#!/usr/bin/env python
"""
MR-CLOVER: MRI CLinical resOlution brain VolumEtRics

A streamlined pipeline for brain extraction and tissue segmentation from clinical MRI.
Uses SynthSeg as the core segmentation engine - works with any MRI contrast.

Pipeline:
    1. Bias field correction pass 1 (ANTs N4, unmasked)
    2. SynthSeg segmentation (FreeSurfer)
    3. Estimate WM intensity mode
    4. Build normal-appearing WM mask (FWHM-based, excludes pathology)
    5. Bias field correction pass 2 (ANTs N4, masked to normal WM)
    6. Combine bias fields (B1 * B2)
    7. Derive all outputs from SynthSeg labels:
       - Brain mask (all tissue labels)
       - GMWM mask (grey + white matter labels)
       - ICV mask (intracranial volume)
       - Ventricle mask
       - Optional: intensity-normalized image
       - Optional: combined bias field for on-the-fly correction

Author: MDS
Organization: MGH/HMS
Contact: mschirmer1@mgh.harvard.edu
Version: 0.5.0.0
Date: 2026-11-02

Changes in 0.5.0.0 (SynthSeg-based rewrite, first developed inside WMHP as 0.3.0):
    - N4 pass 2 now uses GMWM mask (more robust for high-res images)
    - Ventricle refinement now uses probabilistic Gaussian classification
    - Tissue distributions estimated from normal-appearing tissue (excludes lesions)
    - Ventricle dilation is in-plane only (avoids through-plane bleeding)
    - Ventricle refinement includes hole filling and connectivity constraint
    - Fixed ANTsPy N4 crash on large images (masks must be float32, not uint8)
    - Mean-shift mode estimation now subsamples large arrays (faster on high-res)
"""

from __future__ import annotations

__version__ = '0.5.0.0'
__author__ = 'mds'

import argparse
import csv
import os
import shutil
import sys
import tempfile
from pathlib import Path
from subprocess import run
from typing import Optional, Tuple

import ants
import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Constants
# =============================================================================

DEFAULT_NORM_TARGET = 0.75
MIN_VOXELS_FOR_MODE = 100
DEFAULT_FWHM_MULTIPLIER = 1.0

SYNTHSEG_LABELS = {
    'wm': [2, 41, 7, 46, 16],  # L/R Cerebral WM, L/R Cerebellar WM, Brain Stem
    'gm_cortical': [3, 42, 8, 47],  # L/R Cerebral Cortex, L/R Cerebellar Cortex
    'gm_subcortical': [
        10, 49,  # L/R Thalamus
        11, 50,  # L/R Caudate
        12, 51,  # L/R Putamen
        13, 52,  # L/R Pallidum
        17, 53,  # L/R Hippocampus
        18, 54,  # L/R Amygdala
        26, 58,  # L/R Accumbens
        28, 60,  # L/R Ventral DC
    ],
    'csf': [4, 43, 5, 44, 14, 15, 24],  # Ventricles and CSF
    'ventricles': [4, 43, 5, 44, 14, 15],  # L/R Lateral, L/R Inf Lateral, 3rd, 4th
}


# =============================================================================
# Tissue Distribution Dataclass
# =============================================================================

from dataclasses import dataclass

@dataclass
class TissueDistributions:
    """Holds intensity distribution parameters for tissue classes."""
    wm_mode: Optional[float] = None
    wm_std: Optional[float] = None
    gm_mode: Optional[float] = None
    gm_std: Optional[float] = None
    csf_mode: Optional[float] = None
    csf_std: Optional[float] = None
    
    def is_complete(self) -> bool:
        """Check if all distributions are available."""
        return all([
            self.wm_mode is not None, self.wm_std is not None,
            self.gm_mode is not None, self.gm_std is not None,
            self.csf_mode is not None, self.csf_std is not None
        ])


# =============================================================================
# Label Helpers
# =============================================================================

def get_gm_labels() -> list[int]:
    """Get all grey matter labels."""
    return SYNTHSEG_LABELS['gm_cortical'] + SYNTHSEG_LABELS['gm_subcortical']


def get_wm_labels() -> list[int]:
    """Get all white matter labels."""
    return SYNTHSEG_LABELS['wm']


def get_all_tissue_labels() -> list[int]:
    """Get all GM + WM labels (brain parenchyma)."""
    return get_wm_labels() + get_gm_labels()


def get_all_brain_labels() -> list[int]:
    """Get all intracranial labels (tissue + CSF)."""
    return get_all_tissue_labels() + SYNTHSEG_LABELS['csf']


# =============================================================================
# Logging
# =============================================================================

class Log:
    """Simple color-coded logging."""
    
    COLORS = {
        'error': '\033[91m',
        'warn': '\033[93m',
        'ok': '\033[92m',
        'info': '\033[94m',
        'debug': '\033[95m',
        'reset': '\033[0m'
    }
    
    verbose: bool = False
    debug_mode: bool = False
    
    @classmethod
    def error(cls, msg: str) -> None:
        print(f"{cls.COLORS['error']}[ERROR] {msg}{cls.COLORS['reset']}")
    
    @classmethod
    def warn(cls, msg: str) -> None:
        print(f"{cls.COLORS['warn']}[WARN] {msg}{cls.COLORS['reset']}")
    
    @classmethod
    def ok(cls, msg: str) -> None:
        print(f"{cls.COLORS['ok']}[OK] {msg}{cls.COLORS['reset']}")
    
    @classmethod
    def info(cls, msg: str) -> None:
        if cls.verbose:
            print(f"{cls.COLORS['info']}[INFO] {msg}{cls.COLORS['reset']}")
    
    @classmethod
    def dbg(cls, msg: str) -> None:
        if cls.debug_mode:
            print(f"{cls.COLORS['debug']}[DEBUG] {msg}{cls.COLORS['reset']}")


# =============================================================================
# Mask Extraction
# =============================================================================

def extract_mask(segmentation: NDArray, labels: list[int]) -> NDArray[np.uint8]:
    """Create binary mask from segmentation using specified labels."""
    return np.isin(segmentation, labels).astype(np.uint8)


def extract_brain_mask(segmentation: NDArray) -> NDArray[np.uint8]:
    """Extract brain mask (all intracranial structures)."""
    return extract_mask(segmentation, get_all_brain_labels())


def extract_gmwm_mask(segmentation: NDArray) -> NDArray[np.uint8]:
    """Extract grey/white matter mask (brain parenchyma, no CSF)."""
    return extract_mask(segmentation, get_all_tissue_labels())


def extract_icv_mask(segmentation: NDArray) -> NDArray[np.uint8]:
    """Extract intracranial volume mask."""
    return extract_brain_mask(segmentation)


def extract_ventricle_mask(segmentation: NDArray) -> NDArray[np.uint8]:
    """Extract ventricle mask."""
    return extract_mask(segmentation, SYNTHSEG_LABELS['ventricles'])


def refine_ventricle_mask(
    image: NDArray,
    ventricle_mask: NDArray,
    distributions: TissueDistributions,
    spacing: Optional[Tuple[float, ...]] = None,
    dilate_iterations: int = 2
) -> NDArray[np.uint8]:
    """
    Refine ventricle mask using probabilistic intensity classification.
    
    Uses Gaussian probability models for CSF, WM, and GM to classify voxels.
    Removes tissue-like voxels from ventricles and adds adjacent CSF-like voxels.
    Dilation is performed in-plane only to avoid extending into adjacent
    slices with coarse through-plane resolution.
    
    Parameters
    ----------
    image
        Bias-corrected image array
    ventricle_mask
        Initial ventricle mask from SynthSeg
    distributions
        TissueDistributions with mode and std for CSF, WM, GM
    spacing
        Voxel spacing (used to determine through-plane axis)
    dilate_iterations
        Number of dilation iterations for expanding search region
        
    Returns
    -------
    NDArray[np.uint8]
        Refined ventricle mask
    """
    from scipy.ndimage import binary_dilation, label
    
    if np.sum(ventricle_mask) == 0:
        return ventricle_mask
    
    # Check if we have complete distributions
    if not distributions.is_complete():
        Log.warn("Incomplete tissue distributions, skipping ventricle refinement")
        return ventricle_mask
    
    # Determine through-plane axis (largest spacing = through-plane)
    if spacing is not None:
        through_plane_axis = np.argmax(np.array(spacing))
    else:
        through_plane_axis = 2  # Default to axial (z)
    
    # Create in-plane structuring element (no dilation in through-plane direction)
    struct = np.ones((3, 3, 3), dtype=bool)
    if through_plane_axis == 0:
        struct[0, :, :] = False
        struct[2, :, :] = False
    elif through_plane_axis == 1:
        struct[:, 0, :] = False
        struct[:, 2, :] = False
    else:  # axis == 2
        struct[:, :, 0] = False
        struct[:, :, 2] = False
    
    Log.info(f"Ventricle refinement using Gaussian classification:")
    Log.info(f"  CSF: mode={distributions.csf_mode:.1f}, std={distributions.csf_std:.1f}")
    Log.info(f"  WM:  mode={distributions.wm_mode:.1f}, std={distributions.wm_std:.1f}")
    Log.info(f"  GM:  mode={distributions.gm_mode:.1f}, std={distributions.gm_std:.1f}")
    
    # Compute Gaussian log-probabilities (avoiding underflow)
    # log p(x|class) = -0.5 * ((x - mu) / sigma)^2 - log(sigma)
    def log_gaussian_prob(x: NDArray, mu: float, sigma: float) -> NDArray:
        return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma)
    
    log_p_csf = log_gaussian_prob(image, distributions.csf_mode, distributions.csf_std)
    log_p_wm = log_gaussian_prob(image, distributions.wm_mode, distributions.wm_std)
    log_p_gm = log_gaussian_prob(image, distributions.gm_mode, distributions.gm_std)
    
    # CSF-like if CSF is most probable
    is_csf_like = (log_p_csf > log_p_wm) & (log_p_csf > log_p_gm)
    
    # Step 1: Remove tissue-like voxels from ventricles
    refined = (ventricle_mask > 0) & is_csf_like
    n_removed = np.sum(ventricle_mask) - np.sum(refined)
    
    if n_removed > 0:
        Log.info(f"Removed {n_removed} tissue-like voxels from ventricles")
    
    # Step 2: Add adjacent CSF-like voxels (in-plane dilation only)
    dilated = binary_dilation(refined, structure=struct, iterations=dilate_iterations)
    expanded = dilated & is_csf_like
    
    n_added = np.sum(expanded) - np.sum(refined)
    if n_added > 0:
        Log.info(f"Added {n_added} adjacent CSF voxels to ventricles (in-plane)")
    
    # Step 3: Fill holes slice-by-slice (in-plane)
    filled = _fill_holes_slice_by_slice(expanded, through_plane_axis)
    n_filled = np.sum(filled) - np.sum(expanded)
    if n_filled > 0:
        Log.info(f"Filled {n_filled} hole voxels in ventricles")
    
    # Step 4: Keep only voxels connected to original ventricle mask
    labeled, n_components = label(filled)
    
    if n_components > 1:
        # Find which labels overlap with original ventricle mask
        original_labels = set(np.unique(labeled[ventricle_mask > 0]))
        original_labels.discard(0)  # Remove background
        
        # Keep only connected components that touch original ventricles
        connected_mask = np.isin(labeled, list(original_labels))
        n_disconnected = np.sum(filled) - np.sum(connected_mask)
        
        if n_disconnected > 0:
            Log.info(f"Removed {n_disconnected} disconnected voxels from ventricles")
        
        return connected_mask.astype(np.uint8)
    
    return filled.astype(np.uint8)


# =============================================================================
# Mean-Shift Mode Finding
# =============================================================================

# Maximum samples for mode estimation (larger arrays are subsampled)
MAX_SAMPLES_FOR_MODE = 100000


def mean_shift_mode(
    data: NDArray,
    sigma: Optional[float] = None,
    n_replicates: int = 10,
    epsilon: Optional[float] = None,
    max_iter: int = 1000,
    max_samples: int = MAX_SAMPLES_FOR_MODE
) -> float:
    """
    Find the mode of a distribution using mean-shift algorithm.
    
    More robust than median/mean for finding the dominant peak in 
    potentially skewed or multimodal distributions.
    
    Parameters
    ----------
    data
        Input data array
    sigma
        Kernel bandwidth (auto-computed if None)
    n_replicates
        Number of starting points
    epsilon
        Convergence threshold
    max_iter
        Maximum iterations per replicate
    max_samples
        Maximum samples to use (subsamples if larger)
    """
    data = data.flatten().astype(np.float64)
    
    if len(data) == 0:
        return 0.0
    
    # Subsample if too large - mode estimation doesn't need all points
    if len(data) > max_samples:
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        data = rng.choice(data, max_samples, replace=False)
    
    # Bandwidth selection (Silverman's rule of thumb with MAD)
    if sigma is None:
        mad = np.median(np.abs(data - np.median(data)))
        sigma = mad / 0.6745 * (4.0 / 3.0 / len(data)) ** 0.2
        sigma = max(sigma, 1e-6)
    
    if epsilon is None:
        epsilon = sigma / 100.0
    
    # Initialize from percentiles for coverage
    percentiles = np.linspace(10, 90, n_replicates)
    inits = np.percentile(data, percentiles)
    
    best_mode = np.median(data)
    best_score = -np.inf
    
    # Precompute histogram for scoring
    n_bins = max(100, len(data) // 100)
    hist, bin_edges = np.histogram(data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    for init in inits:
        mode = init
        
        for _ in range(max_iter):
            weights = np.exp(-0.5 * ((data - mode) / sigma) ** 2)
            weight_sum = weights.sum()
            
            if weight_sum < 1e-10:
                break
            
            new_mode = np.dot(weights, data) / weight_sum
            
            if abs(new_mode - mode) < epsilon:
                mode = new_mode
                break
            
            mode = new_mode
        
        # Score using kernel density
        kernel = np.exp(-0.5 * ((bin_centers - mode) / sigma) ** 2)
        score = np.dot(kernel, hist)
        
        if score > best_score:
            best_score = score
            best_mode = mode
    
    return best_mode


# =============================================================================
# Normal-Appearing WM Mask (FWHM-based)
# =============================================================================

def extract_normal_wm_mask_fwhm(
    image: NDArray,
    segmentation: NDArray,
    wm_mode: float,
    fwhm_multiplier: float = DEFAULT_FWHM_MULTIPLIER
) -> NDArray[np.uint8]:
    """
    Extract normal-appearing WM using FWHM of intensity distribution.
    
    Excludes pathology (WMH, stroke lesions) that have intensities
    outside the main WM peak.
    
    Parameters
    ----------
    image
        Bias-corrected image array
    segmentation
        SynthSeg segmentation array
    wm_mode
        Mode intensity of WM distribution
    fwhm_multiplier
        Multiplier for FWHM width (1.0 = standard FWHM, >1 = wider)
        
    Returns
    -------
    NDArray[np.uint8]
        Binary mask of normal-appearing WM
    """
    from scipy.ndimage import binary_erosion
    
    wm_seg = extract_mask(segmentation, get_wm_labels())
    wm_seg_eroded = binary_erosion(wm_seg, iterations=2)
    if np.sum(wm_seg_eroded) < MIN_VOXELS_FOR_MODE:
        wm_seg_eroded = wm_seg
    
    wm_intensities = image[wm_seg_eroded > 0]
    
    if len(wm_intensities) < MIN_VOXELS_FOR_MODE:
        Log.warn("Insufficient WM voxels for FWHM estimation, using all WM")
        return wm_seg
    
    # Estimate σ from MAD (robust to outliers like WMH)
    mad = np.median(np.abs(wm_intensities - wm_mode))
    sigma = mad / 0.6745
    
    # FWHM = 2.355 * σ, so half-width = 1.1775 * σ
    half_width = 1.1775 * sigma * fwhm_multiplier
    
    lower = wm_mode - half_width
    upper = wm_mode + half_width
    
    Log.info(f"Normal WM intensity range: [{lower:.1f}, {upper:.1f}] (mode={wm_mode:.1f}, σ={sigma:.1f})")
    
    intensity_ok = (image >= lower) & (image <= upper)
    normal_wm = (wm_seg > 0) & intensity_ok
    
    n_total = np.sum(wm_seg)
    n_normal = np.sum(normal_wm)
    n_excluded = n_total - n_normal
    
    Log.info(f"Normal WM: {n_normal} voxels, excluded {n_excluded} ({100*n_excluded/n_total:.1f}%) as pathology")
    
    return normal_wm.astype(np.uint8)


def compute_wm_mode(
    image: NDArray,
    segmentation: NDArray
) -> Optional[float]:
    """
    Compute WM intensity mode from eroded WM mask.
    
    Parameters
    ----------
    image
        Bias-corrected image array
    segmentation
        SynthSeg segmentation array
        
    Returns
    -------
    Optional[float]
        WM mode intensity, or None if insufficient voxels
    """
    from scipy.ndimage import binary_erosion
    
    wm_seg = extract_mask(segmentation, get_wm_labels())
    wm_seg_eroded = binary_erosion(wm_seg, iterations=2)
    
    if np.sum(wm_seg_eroded) < MIN_VOXELS_FOR_MODE:
        wm_seg_eroded = wm_seg
    
    wm_intensities = image[wm_seg_eroded > 0]
    
    if len(wm_intensities) < MIN_VOXELS_FOR_MODE:
        Log.warn("Insufficient WM voxels for mode estimation")
        return None
    
    wm_mode = mean_shift_mode(wm_intensities)
    Log.info(f"WM mode intensity: {wm_mode:.1f}")
    
    return wm_mode


def estimate_tissue_distributions(
    image: NDArray,
    segmentation: NDArray,
    fwhm_multiplier: float = 1.5
) -> TissueDistributions:
    """
    Estimate intensity distributions for WM, GM, and CSF from normal-appearing tissue.
    
    Uses FWHM-based outlier exclusion to identify normal tissue, avoiding
    contamination from lesions.
    
    Parameters
    ----------
    image
        Bias-corrected image array
    segmentation
        SynthSeg segmentation array
    fwhm_multiplier
        Multiplier for FWHM width when identifying normal tissue (>1 = more inclusive)
        
    Returns
    -------
    TissueDistributions
        Dataclass with mode and std for each tissue class
    """
    from scipy.ndimage import binary_erosion
    
    distributions = TissueDistributions()
    
    # Helper to compute normal-appearing distribution
    def compute_normal_distribution(
        mask: NDArray, 
        tissue_name: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute mode and std from normal-appearing voxels within mask."""
        # Erode to avoid partial volume
        mask_eroded = binary_erosion(mask, iterations=2)
        if np.sum(mask_eroded) < MIN_VOXELS_FOR_MODE:
            mask_eroded = mask
        
        intensities = image[mask_eroded > 0]
        if len(intensities) < MIN_VOXELS_FOR_MODE:
            Log.warn(f"Insufficient {tissue_name} voxels for distribution estimation")
            return None, None
        
        # Initial mode estimate
        mode = mean_shift_mode(intensities)
        
        # MAD-based sigma (robust to outliers)
        mad = np.median(np.abs(intensities - mode))
        sigma_robust = mad / 0.6745
        
        # Define normal range using FWHM
        half_width = 1.1775 * sigma_robust * fwhm_multiplier
        lower = mode - half_width
        upper = mode + half_width
        
        # Get normal-appearing voxels
        normal_mask = mask_eroded & (image >= lower) & (image <= upper)
        normal_intensities = image[normal_mask > 0]
        
        if len(normal_intensities) < MIN_VOXELS_FOR_MODE:
            Log.warn(f"Insufficient normal {tissue_name} voxels, using all")
            normal_intensities = intensities
        
        # Recompute mode and std from normal-appearing tissue
        final_mode = mean_shift_mode(normal_intensities)
        final_std = np.std(normal_intensities)
        
        n_total = np.sum(mask_eroded)
        n_normal = len(normal_intensities)
        n_excluded = n_total - n_normal
        
        if n_excluded > 0:
            Log.info(f"Normal {tissue_name}: excluded {n_excluded} ({100*n_excluded/n_total:.1f}%) outlier voxels")
        Log.info(f"{tissue_name} distribution: mode={final_mode:.1f}, std={final_std:.1f}")
        
        return final_mode, final_std
    
    # WM distribution
    wm_mask = extract_mask(segmentation, get_wm_labels())
    distributions.wm_mode, distributions.wm_std = compute_normal_distribution(wm_mask, "WM")
    
    # GM distribution (cortical + subcortical)
    gm_mask = extract_mask(segmentation, get_gm_labels())
    distributions.gm_mode, distributions.gm_std = compute_normal_distribution(gm_mask, "GM")
    
    # CSF distribution (from ventricles - should be clean)
    csf_mask = extract_mask(segmentation, SYNTHSEG_LABELS['ventricles'])
    distributions.csf_mode, distributions.csf_std = compute_normal_distribution(csf_mask, "CSF")
    
    return distributions


# =============================================================================
# GMWM Refinement with Intensity Analysis
# =============================================================================

def extract_gmwm_refined(
    image: NDArray,
    segmentation: NDArray,
    spacing: Optional[Tuple[float, ...]] = None
) -> Tuple[NDArray[np.uint8], Optional[float], Optional[float], Optional[float], NDArray[np.uint8]]:
    """
    Extract GMWM mask using intensity refinement with ICV recovery.
    
    Returns
    -------
    tuple
        (gmwm_mask, gm_mode, wm_mode, csf_mode, icv_mask)
    """
    from scipy.ndimage import binary_erosion, binary_fill_holes, gaussian_filter
    
    # Create seed masks
    gm_seed = extract_mask(segmentation, get_gm_labels())
    wm_seed = extract_mask(segmentation, get_wm_labels())
    csf_seed = extract_mask(segmentation, SYNTHSEG_LABELS['csf'])
    tissue_seed = extract_mask(segmentation, get_all_tissue_labels())
    
    # Erode seeds to avoid partial volume
    gm_seed_eroded = binary_erosion(gm_seed, iterations=2)
    wm_seed_eroded = binary_erosion(wm_seed, iterations=2)
    tissue_seed_eroded = binary_erosion(tissue_seed, iterations=2)
    csf_seed_eroded = binary_erosion(csf_seed, iterations=2)
    
    # Fall back to uneroded if too small
    if np.sum(gm_seed_eroded) < MIN_VOXELS_FOR_MODE:
        gm_seed_eroded = gm_seed
    if np.sum(wm_seed_eroded) < MIN_VOXELS_FOR_MODE:
        wm_seed_eroded = wm_seed
    if np.sum(tissue_seed_eroded) < MIN_VOXELS_FOR_MODE:
        tissue_seed_eroded = tissue_seed
    if np.sum(csf_seed_eroded) < MIN_VOXELS_FOR_MODE:
        csf_seed_eroded = csf_seed
    
    # Get intensities
    gm_intensities = image[gm_seed_eroded > 0]
    wm_intensities = image[wm_seed_eroded > 0]
    tissue_intensities = image[tissue_seed_eroded > 0]
    csf_intensities = image[csf_seed_eroded > 0]
    
    # Check sample sizes
    if len(tissue_intensities) < MIN_VOXELS_FOR_MODE or len(csf_intensities) < MIN_VOXELS_FOR_MODE:
        Log.warn("Insufficient voxels for refinement, using direct SynthSeg extraction")
        return extract_gmwm_mask(segmentation), None, None, None, extract_icv_mask(segmentation)
    
    # Compute modes
    gm_mode = mean_shift_mode(gm_intensities) if len(gm_intensities) >= MIN_VOXELS_FOR_MODE else None
    wm_mode = mean_shift_mode(wm_intensities) if len(wm_intensities) >= MIN_VOXELS_FOR_MODE else None
    tissue_mode = mean_shift_mode(tissue_intensities)
    csf_mode = mean_shift_mode(csf_intensities)
    
    gm_str = f"{gm_mode:.1f}" if gm_mode is not None else "N/A"
    wm_str = f"{wm_mode:.1f}" if wm_mode is not None else "N/A"
    Log.info(f"Intensity modes - GM: {gm_str}, WM: {wm_str}, CSF: {csf_mode:.1f}")
    
    # ICV hole filling
    brain_mask = segmentation > 0
    through_plane_axis = np.argmax(np.array(spacing)) if spacing else 2
    
    filled_icv = _fill_holes_slice_by_slice(brain_mask, through_plane_axis)
    recovered_mask = filled_icv & ~brain_mask
    n_recovered = np.sum(recovered_mask)
    
    if n_recovered > 0:
        Log.info(f"ICV hole-filling recovered {n_recovered} voxels")
    
    # In-plane smoothing
    image_smooth = _smooth_in_plane(image, spacing)
    
    # Global classification
    gmwm_mask = _classify_brain_voxels(image_smooth, brain_mask, tissue_mode, csf_mode)
    
    # Subcortical refinement
    gmwm_mask = _refine_subcortical(
        gmwm_mask, image, image_smooth, segmentation, tissue_mode, csf_mode
    )
    
    # Classify recovered voxels
    if n_recovered > 0:
        gmwm_mask = _classify_recovered_voxels(
            gmwm_mask, image_smooth, recovered_mask, tissue_mode, csf_mode
        )
    
    # ICV = hole-filled union GMWM
    icv_mask = (filled_icv | gmwm_mask).astype(np.uint8)
    
    return gmwm_mask, gm_mode, wm_mode, csf_mode, icv_mask


def _fill_holes_slice_by_slice(mask: NDArray, axis: int) -> NDArray:
    """Fill holes in each 2D slice along the given axis."""
    from scipy.ndimage import binary_fill_holes
    
    filled = np.zeros_like(mask)
    n_slices = mask.shape[axis]
    
    for i in range(n_slices):
        if axis == 0:
            slice_2d = mask[i, :, :]
            filled[i, :, :] = binary_fill_holes(slice_2d)
        elif axis == 1:
            slice_2d = mask[:, i, :]
            filled[:, i, :] = binary_fill_holes(slice_2d)
        else:
            slice_2d = mask[:, :, i]
            filled[:, :, i] = binary_fill_holes(slice_2d)
    
    return filled


def _smooth_in_plane(image: NDArray, spacing: Optional[Tuple[float, ...]]) -> NDArray:
    """Apply in-plane Gaussian smoothing."""
    from scipy.ndimage import gaussian_filter
    
    if spacing is None:
        sigma = (1, 1, 0)  # Default axial
    else:
        through_plane_axis = np.argmax(np.array(spacing))
        sigma = [1, 1, 1]
        sigma[through_plane_axis] = 0
        sigma = tuple(sigma)
    
    return gaussian_filter(image, sigma=sigma)


def _classify_brain_voxels(
    image_smooth: NDArray,
    brain_mask: NDArray,
    tissue_mode: float,
    csf_mode: float
) -> NDArray[np.uint8]:
    """Classify brain voxels as tissue vs CSF based on intensity."""
    brain_voxels = image_smooth[brain_mask]
    
    dist_to_tissue = np.abs(brain_voxels - tissue_mode)
    dist_to_csf = np.abs(brain_voxels - csf_mode)
    is_tissue = dist_to_tissue <= dist_to_csf
    
    gmwm_mask = np.zeros_like(brain_mask, dtype=np.uint8)
    brain_indices = np.where(brain_mask)
    gmwm_mask[brain_indices[0][is_tissue],
              brain_indices[1][is_tissue],
              brain_indices[2][is_tissue]] = 1
    
    return gmwm_mask


def _refine_subcortical(
    gmwm_mask: NDArray[np.uint8],
    image: NDArray,
    image_smooth: NDArray,
    segmentation: NDArray,
    tissue_mode: float,
    csf_mode: float
) -> NDArray[np.uint8]:
    """Refine classification for subcortical structures."""
    subcortical_labels = SYNTHSEG_LABELS['gm_subcortical']
    subcortical_mask = extract_mask(segmentation, subcortical_labels)
    suspicious_mask = (subcortical_mask & (gmwm_mask == 0)).astype(bool)
    n_suspicious = np.sum(suspicious_mask)
    
    if n_suspicious == 0:
        return gmwm_mask
    
    Log.info(f"Refining {n_suspicious} suspicious subcortical voxels...")
    
    # Compute per-structure medians
    labels_to_refine = set(np.unique(segmentation[suspicious_mask])) & set(subcortical_labels)
    label_modes = {}
    for label in labels_to_refine:
        label_voxels = image[segmentation == label]
        if len(label_voxels) >= MIN_VOXELS_FOR_MODE:
            label_modes[label] = np.median(label_voxels)
        else:
            label_modes[label] = tissue_mode
    
    # Re-classify
    susp_voxels = image_smooth[suspicious_mask]
    susp_labels = segmentation[suspicious_mask]
    susp_modes = np.array([label_modes[l] for l in susp_labels])
    
    dist_to_struct = np.abs(susp_voxels - susp_modes)
    dist_to_csf = np.abs(susp_voxels - csf_mode)
    is_tissue = dist_to_struct <= dist_to_csf
    
    susp_idx = np.where(suspicious_mask)
    gmwm_mask[susp_idx[0][is_tissue],
              susp_idx[1][is_tissue],
              susp_idx[2][is_tissue]] = 1
    
    Log.info(f"Reclassified {np.sum(is_tissue)}/{n_suspicious} as tissue")
    
    return gmwm_mask


def _classify_recovered_voxels(
    gmwm_mask: NDArray[np.uint8],
    image_smooth: NDArray,
    recovered_mask: NDArray,
    tissue_mode: float,
    csf_mode: float
) -> NDArray[np.uint8]:
    """Classify voxels recovered by hole filling."""
    recovered_voxels = image_smooth[recovered_mask]
    
    dist_to_tissue = np.abs(recovered_voxels - tissue_mode)
    dist_to_csf = np.abs(recovered_voxels - csf_mode)
    is_tissue = dist_to_tissue <= dist_to_csf
    
    recovered_idx = np.where(recovered_mask)
    gmwm_mask[recovered_idx[0][is_tissue],
              recovered_idx[1][is_tissue],
              recovered_idx[2][is_tissue]] = 1
    
    n_tissue = np.sum(is_tissue)
    Log.info(f"Recovered voxels classified as tissue: {n_tissue}/{np.sum(recovered_mask)}")
    
    return gmwm_mask


# =============================================================================
# Intensity Normalization
# =============================================================================

def normalize_intensity(
    image: NDArray,
    tissue_mask: NDArray,
    target: float = DEFAULT_NORM_TARGET
) -> Tuple[NDArray, float]:
    """
    Normalize image intensity based on tissue intensity mode.
    
    Returns
    -------
    tuple
        (normalized_image, mode_intensity)
    """
    tissue_values = image[tissue_mask > 0]
    
    if len(tissue_values) == 0:
        Log.warn("No tissue voxels found for normalization")
        return image, 1.0
    
    mode_intensity = mean_shift_mode(tissue_values)
    
    if mode_intensity <= 0:
        Log.warn(f"Invalid mode intensity ({mode_intensity}), skipping normalization")
        return image, 1.0
    
    factor = target / mode_intensity
    normalized = image * factor
    
    Log.info(f"Intensity normalization: mode={mode_intensity:.2f}, factor={factor:.4f}")
    
    return normalized, mode_intensity


# =============================================================================
# Utility Functions for On-the-fly Processing
# =============================================================================

def read_stats_csv(stats_path: str | Path) -> dict[str, str]:
    """Read MR-Clover stats CSV and return as dict."""
    with open(stats_path) as f:
        reader = csv.DictReader(f)
        return next(reader)


def get_wm_mode_from_stats(stats_path: str | Path) -> float:
    """Extract WM mode intensity from stats CSV."""
    stats = read_stats_csv(stats_path)
    
    if 'wm_mode_intensity' in stats:
        return float(stats['wm_mode_intensity'])
    elif 'tissue_mode_intensity' in stats:
        return float(stats['tissue_mode_intensity'])
    else:
        raise ValueError(f"No WM or tissue mode found in {stats_path}")


def apply_bias_correction(
    input_path: str | Path,
    bias_field_path: str | Path,
    output_path: Optional[str | Path] = None
) -> ants.ANTsImage:
    """
    Apply bias field correction only (no intensity normalization).
    
    Corrected = Input / BiasField
    """
    img = ants.image_read(str(input_path))
    bias_field = ants.image_read(str(bias_field_path))
    
    img_data = img.numpy()
    bias_data = bias_field.numpy()
    
    # Avoid division by zero
    bias_data = np.where(bias_data > 0, bias_data, 1.0)
    corrected_data = img_data / bias_data
    
    corrected = img.new_image_like(corrected_data)
    
    if output_path:
        corrected.to_filename(str(output_path))
        Log.ok(f"Saved bias-corrected image: {output_path}")
    
    return corrected


def apply_bias_and_normalize(
    input_path: str | Path,
    bias_field_path: str | Path,
    stats_csv_path: str | Path,
    output_path: Optional[str | Path] = None,
    target: float = DEFAULT_NORM_TARGET
) -> ants.ANTsImage:
    """
    Apply bias field correction and WM intensity normalization.
    
    Parameters
    ----------
    input_path
        Path to raw input image (before bias correction)
    bias_field_path
        Path to bias field image from MR-Clover
    stats_csv_path
        Path to MR-Clover stats CSV (contains wm_mode_intensity)
    output_path
        If provided, save result to this path
    target
        Target intensity for WM (default: 0.75)
        
    Returns
    -------
    ants.ANTsImage
        Bias-corrected and WM-normalized image
    """
    # Apply bias correction
    corrected = apply_bias_correction(input_path, bias_field_path)
    
    # Get WM peak from CSV
    wm_peak = get_wm_mode_from_stats(stats_csv_path)
    
    if wm_peak <= 0:
        raise ValueError(f"Invalid WM peak value: {wm_peak}")
    
    # Normalize
    factor = target / wm_peak
    normalized_data = corrected.numpy() * factor
    normalized = corrected.new_image_like(normalized_data)
    
    if output_path:
        normalized.to_filename(str(output_path))
        Log.ok(f"Saved normalized image: {output_path}")
    
    return normalized


# =============================================================================
# SynthSeg
# =============================================================================

def run_synthseg(
    input_file: str,
    output_file: str,
    use_cpu: bool = False,
    robust: bool = False
) -> bool:
    """Run SynthSeg segmentation."""
    cmd = ['mri_synthseg', '--i', input_file, '--o', output_file, '--keepgeom']
    
    if robust:
        cmd.append('--robust')
        Log.info("Using robust mode (slower)")
    
    if use_cpu:
        cmd.extend(['--cpu', '--threads', str(max(1, os.cpu_count() - 1))])
    
    Log.info(f"Running SynthSeg: {' '.join(cmd)}")
    
    try:
        result = run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            Log.error(f"SynthSeg failed: {result.stderr}")
            return False
        return True
    except FileNotFoundError:
        Log.error("mri_synthseg not found. Please install FreeSurfer.")
        return False
    except Exception as e:
        Log.error(f"SynthSeg error: {e}")
        return False


# =============================================================================
# Bias Field Correction (Two-Pass)
# =============================================================================

def run_n4_bias_correction(
    img: ants.ANTsImage,
    mask: Optional[ants.ANTsImage] = None
) -> Tuple[ants.ANTsImage, ants.ANTsImage]:
    """
    Run N4 bias field correction and return both corrected image and bias field.
    
    Parameters
    ----------
    img
        Input ANTs image
    mask
        Optional mask to constrain bias field estimation
    
    Returns
    -------
    tuple
        (bias_corrected_image, bias_field)
    """
    # Get bias field (return_bias_field=True returns field instead of corrected)
    if mask is not None:
        # ANTsPy N4 crashes with uint8 masks on large images - must use float32
        mask_float = mask.new_image_like(mask.numpy().astype(np.float32))
        bias_field = ants.n4_bias_field_correction(img, mask=mask_float, return_bias_field=True)
    else:
        bias_field = ants.n4_bias_field_correction(img, return_bias_field=True)
    
    # Compute corrected image: corrected = input / bias_field
    img_data = img.numpy()
    bias_data = bias_field.numpy()
    bias_data = np.where(bias_data > 0, bias_data, 1.0)
    corrected_data = img_data / bias_data
    
    img_corrected = img.new_image_like(corrected_data)
    
    return img_corrected, bias_field


def combine_bias_fields(
    bias_field_1: ants.ANTsImage,
    bias_field_2: ants.ANTsImage
) -> ants.ANTsImage:
    """
    Combine two multiplicative bias fields.
    
    Since corrected = raw / B1 / B2 = raw / (B1 * B2),
    the combined bias field is the product.
    
    Parameters
    ----------
    bias_field_1
        First bias field (from unmasked N4)
    bias_field_2
        Second bias field (from masked N4)
        
    Returns
    -------
    ants.ANTsImage
        Combined bias field (B1 * B2)
    """
    combined_data = bias_field_1.numpy() * bias_field_2.numpy()
    return bias_field_1.new_image_like(combined_data)


# =============================================================================
# Statistics Writing
# =============================================================================

def write_stats_csv(stats: list[tuple[str, str]], path: str | Path) -> None:
    """Write statistics to CSV file."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = [s[0] for s in stats]
        values = [s[1] for s in stats]
        writer.writerow(headers)
        writer.writerow(values)


# =============================================================================
# Main Pipeline
# =============================================================================

def process(args: argparse.Namespace) -> int:
    """Main processing pipeline."""
    Log.verbose = args.verbose
    Log.debug_mode = args.debug
    
    if not os.path.isfile(args.input):
        Log.error(f"Input file not found: {args.input}")
        return 1
    
    Log.ok(f"Processing: {args.input}")
    
    # Create temp directory near outputs (not /tmp for PHI)
    output_paths = [
        args.output, args.brain, args.icv, args.ventricles,
        args.synthseg, args.norm, args.bias, args.bias_field, args.stats
    ]
    output_dir = next((os.path.dirname(os.path.abspath(p)) for p in output_paths if p), None)
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(args.input))
    
    tmpdir = tempfile.mkdtemp(prefix='mrclover_tmp_', dir=output_dir)
    Log.dbg(f"Temp directory: {tmpdir}")
    
    try:
        # Step 1: Load image
        Log.info("Loading input image...")
        img = ants.image_read(args.input)
        voxel_vol = np.prod(img.spacing)
        Log.info(f"Image shape: {img.shape}, spacing: {img.spacing}")
        
        # Handle negative values
        img_data = img.numpy()
        if np.any(img_data < 0):
            Log.info("Adjusting negative intensities")
            img_data = img_data - img_data.min()
            img = img.new_image_like(img_data)
        
        # Step 2: Bias field correction - Pass 1 (unmasked)
        Log.info("Running N4 bias field correction (pass 1: unmasked)...")
        img_pass1, bias_field_1 = run_n4_bias_correction(img)
        
        bias_file = os.path.join(tmpdir, 'bias_corrected_pass1.nii.gz')
        img_pass1.to_filename(bias_file)
        
        # Step 3: SynthSeg
        Log.info("Running SynthSeg segmentation...")
        synthseg_file = args.synthseg if args.synthseg else os.path.join(tmpdir, 'synthseg.nii.gz')
        
        if os.path.isfile(synthseg_file) and not args.force:
            Log.info(f"Using existing SynthSeg output: {synthseg_file}")
        else:
            if not run_synthseg(bias_file, synthseg_file, use_cpu=args.cpu, robust=args.robust):
                return 1
            Log.ok("SynthSeg completed")
        
        if args.synthseg and synthseg_file != args.synthseg:
            shutil.copy(synthseg_file, args.synthseg)
            Log.ok(f"Saved SynthSeg segmentation: {args.synthseg}")
        
        seg_img = ants.image_read(synthseg_file)
        seg_data = seg_img.numpy().astype(int)
        ants.copy_image_info(img, seg_img)
        
        # Step 4: Compute WM mode for second pass
        Log.info("Computing WM intensity mode...")
        wm_mode_pass1 = compute_wm_mode(img_pass1.numpy(), seg_data)
        
        # Step 5: Bias field correction - Pass 2 (masked to brain tissue)
        # Use GMWM mask instead of normal WM mask - more robust for high-res images
        # where FWHM-filtered WM can be fragmented
        if wm_mode_pass1 is not None:
            Log.info("Building GMWM mask for bias correction pass 2...")
            gmwm_mask_pass2 = extract_gmwm_mask(seg_data)
            
            # Convert to ANTs image for N4
            gmwm_ants = img.new_image_like(gmwm_mask_pass2)
            
            Log.info("Running N4 bias field correction (pass 2: GMWM mask)...")
            img_pass2, bias_field_2 = run_n4_bias_correction(img_pass1, mask=gmwm_ants)
            
            # Combine bias fields
            combined_bias_field = combine_bias_fields(bias_field_1, bias_field_2)
            Log.ok("Two-pass bias correction completed")
            
            # Use pass 2 output for downstream processing
            img_corrected = img_pass2
        else:
            Log.warn("Could not compute WM mode, using single-pass bias correction")
            img_corrected = img_pass1
            combined_bias_field = bias_field_1
        
        # Save bias-corrected image if requested
        if args.bias:
            img_corrected.to_filename(args.bias)
            Log.ok(f"Saved bias-corrected image: {args.bias}")
        
        # Save combined bias field if requested
        if args.bias_field:
            combined_bias_field.to_filename(args.bias_field)
            Log.ok(f"Saved combined bias field: {args.bias_field}")
        
        # Step 6: Extract masks
        stats: list[tuple[str, str]] = [
            ('subject', args.subject or os.path.basename(args.input))
        ]
        
        # Brain mask
        if args.brain:
            Log.info("Extracting brain mask...")
            brain_mask = extract_brain_mask(seg_data)
            brain_img = img.new_image_like(brain_mask)
            brain_img.to_filename(args.brain)
            
            brain_vol = voxel_vol * np.sum(brain_mask)
            stats.append(('brain_volume_mm3', f'{brain_vol:.2f}'))
            Log.ok(f"Saved brain mask: {args.brain} ({brain_vol/1000:.1f} cm³)")
        
        # GMWM and ICV
        gm_mode: Optional[float] = None
        wm_mode: Optional[float] = None
        csf_mode: Optional[float] = None
        gmwm_mask: Optional[NDArray] = None
        icv_mask_refined: Optional[NDArray] = None
        tissue_distributions: Optional[TissueDistributions] = None
        
        # Run refinement if needed for GMWM, ICV, or ventricle outputs
        need_refinement = (args.output or args.icv or args.ventricles) and not args.no_refine
        
        if need_refinement:
            Log.info("Extracting GM/WM mask with intensity refinement...")
            try:
                gmwm_mask, gm_mode, wm_mode, csf_mode, icv_mask_refined = extract_gmwm_refined(
                    img_corrected.numpy(), seg_data, spacing=img.spacing
                )
                if gm_mode is not None:
                    stats.append(('gm_mode_intensity', f'{gm_mode:.2f}'))
                if wm_mode is not None:
                    stats.append(('wm_mode_intensity', f'{wm_mode:.2f}'))
                if csf_mode is not None:
                    stats.append(('csf_mode_intensity', f'{csf_mode:.2f}'))
                
                # Estimate distributions from normal-appearing tissue for ventricle refinement
                if args.ventricles:
                    Log.info("Estimating tissue distributions from normal-appearing tissue...")
                    tissue_distributions = estimate_tissue_distributions(
                        img_corrected.numpy(), seg_data
                    )
            except Exception as e:
                Log.warn(f"Refinement failed ({e}), falling back to direct extraction")
                gmwm_mask = extract_gmwm_mask(seg_data)
                icv_mask_refined = extract_icv_mask(seg_data)
        elif args.output or args.icv:
            # No refinement requested, use direct extraction
            Log.info("Extracting GM/WM mask (no refinement)...")
            gmwm_mask = extract_gmwm_mask(seg_data)
            icv_mask_refined = extract_icv_mask(seg_data)
        
        if args.output and gmwm_mask is not None:
            gmwm_img = img.new_image_like(gmwm_mask)
            gmwm_img.to_filename(args.output)
            
            gmwm_vol = voxel_vol * np.sum(gmwm_mask)
            stats.append(('gmwm_volume_mm3', f'{gmwm_vol:.2f}'))
            Log.ok(f"Saved GM/WM mask: {args.output} ({gmwm_vol/1000:.1f} cm³)")
        
        if args.icv and icv_mask_refined is not None:
            icv_img = img.new_image_like(icv_mask_refined)
            icv_img.to_filename(args.icv)
            
            icv_vol = voxel_vol * np.sum(icv_mask_refined)
            stats.append(('icv_volume_mm3', f'{icv_vol:.2f}'))
            Log.ok(f"Saved ICV mask: {args.icv} ({icv_vol/1000:.1f} cm³)")
        
        # Ventricles
        if args.ventricles:
            Log.info("Extracting ventricle mask...")
            vent_mask_raw = extract_ventricle_mask(seg_data)
            
            # Refine using tissue distributions if available
            if tissue_distributions is not None and tissue_distributions.is_complete():
                vent_mask = refine_ventricle_mask(
                    img_corrected.numpy(),
                    vent_mask_raw,
                    distributions=tissue_distributions,
                    spacing=img.spacing
                )
            else:
                Log.info("Skipping ventricle refinement (tissue distributions not available)")
                vent_mask = vent_mask_raw
            
            vent_img = img.new_image_like(vent_mask)
            vent_img.to_filename(args.ventricles)
            
            vent_vol = voxel_vol * np.sum(vent_mask)
            stats.append(('ventricle_volume_mm3', f'{vent_vol:.2f}'))
            Log.ok(f"Saved ventricle mask: {args.ventricles} ({vent_vol/1000:.1f} cm³)")
        
        # Normalized image
        if args.norm:
            Log.info("Normalizing intensity...")
            target = args.norm_target
            
            if wm_mode is not None:
                Log.info(f"Using WM mode from refinement: {wm_mode:.1f}")
                factor = target / wm_mode
                norm_data = img_corrected.numpy() * factor
            else:
                tissue_mask = extract_gmwm_mask(seg_data)
                norm_data, mode_intensity = normalize_intensity(
                    img_corrected.numpy(), tissue_mask, target
                )
                if 'wm_mode_intensity' not in [s[0] for s in stats]:
                    stats.append(('wm_mode_intensity', f'{mode_intensity:.2f}'))
            
            norm_img = img.new_image_like(norm_data)
            norm_img.to_filename(args.norm)
            Log.ok(f"Saved normalized image: {args.norm}")
        
        # Step 7: Save statistics
        if args.stats:
            Log.info("Saving statistics...")
            write_stats_csv(stats, args.stats)
            Log.ok(f"Saved statistics: {args.stats}")
        
        Log.ok("Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        Log.error(f"Pipeline failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
        
    finally:
        if not args.debug:
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            Log.dbg(f"Keeping temp files: {tmpdir}")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='MR-CLOVER v1.3: Brain extraction and tissue segmentation for clinical MRI.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic GM/WM mask
  python mr_clover.py -i scan.nii.gz -o gmwm.nii.gz
  
  # Full pipeline with bias field output
  python mr_clover.py -i scan.nii.gz -o gmwm.nii.gz --brain brain.nii.gz \\
      --icv icv.nii.gz --bias-field bias.nii.gz --stats volumes.csv
  
  # Custom normalization target
  python mr_clover.py -i scan.nii.gz --norm norm.nii.gz --norm-target 1.0
'''
    )
    
    # Required
    parser.add_argument('-i', '--input', required=True, help='Input NIfTI image')
    
    # Outputs
    parser.add_argument('-o', '--output', help='Output GM/WM mask')
    parser.add_argument('--brain', help='Output brain mask')
    parser.add_argument('--icv', help='Output intracranial volume mask')
    parser.add_argument('--ventricles', help='Output ventricle mask')
    parser.add_argument('--synthseg', help='Output/input SynthSeg segmentation')
    parser.add_argument('--norm', help='Output intensity-normalized image')
    parser.add_argument('--bias', help='Output bias-corrected image')
    parser.add_argument('--bias-field', help='Output combined bias field (for on-the-fly correction)')
    parser.add_argument('--stats', help='Output CSV with volume statistics')
    
    # Options
    parser.add_argument('--subject', '-s', help='Subject ID for statistics')
    parser.add_argument('--norm-target', type=float, default=DEFAULT_NORM_TARGET,
                        help=f'Target WM intensity for normalization (default: {DEFAULT_NORM_TARGET})')
    parser.add_argument('--no-refine', action='store_true',
                        help='Skip intensity refinement, use raw SynthSeg labels')
    parser.add_argument('--cpu', action='store_true', help='Force CPU processing')
    parser.add_argument('--robust', action='store_true', help='Use SynthSeg robust mode')
    parser.add_argument('--force', '-f', action='store_true', help='Force reprocessing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    args = parser.parse_args()
    
    # Validate: at least one output required
    outputs = [args.output, args.brain, args.icv, args.ventricles,
               args.synthseg, args.norm, args.bias, args.bias_field, args.stats]
    if not any(outputs):
        parser.error("At least one output must be specified")
    
    return process(args)


if __name__ == '__main__':
    sys.exit(main())
