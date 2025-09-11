#!/usr/bin/env python
"""
:Summary: Extracts the brain from clinical MRI and returns grey/white-matter mask.

:Description: This script performs bias field correction, brain extraction, intensity normalization, 
              and generates masks for brain and intracranial volume. The pipeline is modular - 
              each component can be enabled/disabled based on output requirements.

:Requires: Python, NumPy, scipy, scikit-image, warnings, ANTs (Advanced Normalization Tools), 
           Freesurfer (mri_synthstrip, mri_synthseg)

:TODO: Expand to non-CSF-dark modalities (T2, etc.) by:
       - Running SynthSeg for all modalities to get tissue masks
       - Using WM mask from SynthSeg for robust intensity normalization
       - Using GM, WM, and CSF intensities as seed points for GMM refinement
       - Refining the GM/WM mask using seeded GMM approach

:AUTHOR: MDS
:ORGANIZATION: MGH/HMS
:CONTACT: mschirmer1@mgh.harvard.edu
:SINCE: 2025-09-11
:VERSION: 0.3
"""
#=============================================
# Metadata
#=============================================
__author__ = 'mds'
__contact__ = 'mschirmer1@mgh.harvard.edu'
__copyright__ = ''
__license__ = ''
__date__ = '2025-09-11'
__version__ = '0.3'

#=============================================
# Import statements
#=============================================
import sys
import os
from optparse import OptionParser
import re
import multiprocessing
from subprocess import call, CalledProcessError

import numpy as np
import scipy.ndimage as sn
from scipy.ndimage import binary_fill_holes
import skimage.measure as skm
import warnings
import ants
import uuid

#=============================================
# Color-coded logging functions
#=============================================
class Colors:
    """ANSI color codes for terminal output"""
    ERROR = '\033[91m'    # Red
    WARNING = '\033[93m'  # Yellow  
    SUCCESS = '\033[92m'  # Green
    INFO = '\033[94m'     # Blue
    DEBUG = '\033[95m'    # Magenta
    ENDC = '\033[0m'      # Reset

def log_error(msg):
    """Print error message in red"""
    print(f"{Colors.ERROR}[ERROR] {msg}{Colors.ENDC}")
    
def log_warning(msg):
    """Print warning message in yellow"""
    print(f"{Colors.WARNING}[WARNING] {msg}{Colors.ENDC}")
    
def log_success(msg):
    """Print success message in green"""
    print(f"{Colors.SUCCESS}[SUCCESS] {msg}{Colors.ENDC}")
    
def log_info(msg):
    """Print info message in blue"""
    print(f"{Colors.INFO}[INFO] {msg}{Colors.ENDC}")

def log_debug(msg, debug_mode=False):
    """Print debug message in magenta if debug mode is enabled"""
    if debug_mode:
        print(f"{Colors.DEBUG}[DEBUG] {msg}{Colors.ENDC}")

#=============================================
# System utility functions
#=============================================
def get_optimal_threads():
    """
    Get optimal number of threads for processing.
    Uses 90% of available CPU cores, with a minimum of 1.
    
    Returns
    -------
    int
        Number of threads to use
    """
    max_threads = multiprocessing.cpu_count()
    optimal = max(1, int(max_threads * 0.9))
    log_debug(f"System has {max_threads} cores, using {optimal} threads")
    return optimal

def check_external_tools():
    """
    Check if required external tools are available in the system PATH.
    
    Returns
    -------
    bool
        True if all tools are available, False otherwise
    """
    tools = ['mri_synthstrip', 'mri_synthseg']
    all_available = True
    
    for tool in tools:
        try:
            call(['which', tool], stdout=open(os.devnull, 'wb'))
            log_debug(f"Found {tool}")
        except:
            log_error(f"Required tool '{tool}' not found in PATH")
            all_available = False
    
    return all_available

#=============================================
# Image processing helper functions
#=============================================
def mean_shift_mode_finder(data, sigma=None, n_replicates=10, replication_method='percentiles', 
                          epsilon=None, max_iterations=1000, n_bins=None):
    """
    Finds the mode of data using mean shift algorithm.
    Used for robust intensity normalization.

    Parameters
    ----------
    data : ndarray
        One-dimensional data array to find the mode of
    sigma : float, optional
        Kernel bandwidth; defaults to heuristic calculation
    n_replicates : int, optional
        Number of random initializations
    replication_method : str, optional
        'percentiles' or 'random' for initialization method
    epsilon : float, optional
        Convergence threshold
    max_iterations : int, optional
        Maximum iterations per replicate
    n_bins : int, optional
        Number of histogram bins

    Returns
    -------
    tuple
        (mode_value, score) - the best mode and its kernel density score
    """
    # Calculate bandwidth if not provided
    if sigma is None:
        # Optimal bandwidth suggested by Bowman and Azzalini ('97) p31
        sigma = np.median(np.abs(data-np.median(data))) / .6745 * (4./3./float(data.size))**0.2
    
    if epsilon is None:
        epsilon = sigma / 100.
    
    if n_bins is None:
        n_bins = int(max(data.size / 10., 1))

    # Set up histogram for scoring
    dmin, dmax = data.min(), data.max()
    bins = np.linspace(dmin, dmax, n_bins)
    bin_size = (dmax - dmin) / (n_bins - 1.)
    (data_hist, _) = np.histogram(data, bins)
    bin_centers = bins[:-1] + .5 * bin_size

    # Initialize starting points
    if replication_method == 'percentiles':
        if n_replicates > 1:
            percentiles = np.linspace(0, 100, n_replicates)
        else:
            percentiles = [50]
        inits = [np.percentile(data, p) for p in percentiles]
    elif replication_method == 'random':
        inits = np.random.uniform(data.min(), data.max(), n_replicates)

    scores = np.empty(n_replicates)
    means = np.empty(n_replicates)
    
    # Run mean shift for each initialization
    for i in range(n_replicates):
        mean = inits[i]
        change = np.inf
        
        # Mean shift iterations
        for j in range(max_iterations):
            if change < epsilon:
                break
            
            # Calculate weights using Gaussian kernel
            weights = np.exp(-.5 * ((data - mean)/sigma) ** 2)
            
            if weights.sum() == 0:
                log_error(f"Weights sum to 0; increase sigma (current: {sigma})")
                break
            
            # Update mean
            mean_old = mean
            mean = np.dot(weights, data) / float(weights.sum())
            change = np.abs(mean_old - mean)

        if j >= max_iterations - 1:
            log_warning(f'Mean shift did not converge in replicate {i+1}/{n_replicates}')

        means[i] = mean
        
        # Score using kernel density
        kernel = np.exp(-(bin_centers - mean)**2/(2*sigma**2))
        scores[i] = np.dot(kernel, data_hist)

    # Return best result
    best = np.argmax(scores)
    return (means[best], scores[best])

def get_biggest_connected_component(img):
    """
    Extract the largest connected component from a binary image.
    Typically used to isolate the brain from scattered noise.
    
    Parameters
    ----------
    img : ndarray
        Binary image
        
    Returns
    -------
    ndarray
        Binary mask of the largest connected component
    """
    # Label all connected components
    label_img = skm.label(img)
    
    if label_img.max() == 0:  # No components found
        log_warning("No connected components found in image")
        return img
    
    # Find volumes of each component
    volumes = [np.sum(label_img == label) for label in np.unique(label_img[label_img != 0])]
    brain_label = np.unique(label_img[label_img != 0])[np.argmax(volumes)]
    
    return label_img == brain_label

def rescale(img, mask=None, new_intensity=0.75, mode=None):
    """
    Normalize image intensity based on white matter peak.
    
    Parameters
    ----------
    img : ndarray
        Image data
    mask : ndarray, optional
        Brain mask for normalization region
    new_intensity : float, optional
        Target intensity for white matter
    mode : str, optional
        'percentile' or None (uses mean-shift mode finding)
        
    Returns
    -------
    tuple
        (normalized_image, normalization_factor)
    """
    img = img.astype(np.float32)
    
    # Validate or estimate mask
    if mask is not None:
        if mask.shape != img.shape:
            log_error('Mask shape does not match image shape')
            raise ValueError('Mask is of different shape than image')
    else:
        log_info('Estimating brain mask for intensity normalization')
        prec = np.percentile(img, 5)
        mask = img > prec
    
    brain = np.multiply(img, mask)
    
    # Find normalization factor
    if mode == 'percentile':
        # Use 95th percentile of brain voxels
        norm = np.mean(brain[brain > np.percentile(brain, 5)])
        log_debug(f"Using percentile method: norm factor = {norm}")
    else:
        # Use mean-shift to find white matter peak
        (norm, score) = mean_shift_mode_finder(brain[brain > 0.].flatten())
        log_debug(f"Using mean-shift: norm factor = {norm}, score = {score}")
    
    # Check for valid normalization factor
    if norm <= 0 or np.isnan(norm) or np.isinf(norm):
        log_error(f"Invalid normalization factor: {norm}")
        raise ValueError("Intensity normalization failed")
    
    # Rescale intensity
    img = img * new_intensity / float(norm)
    
    return img, norm

def validate_icv_mask(icv_mask, brain_mask, debug_mode=False):
    """
    Validate ICV mask quality by checking if brain is properly contained.
    
    Parameters
    ----------
    icv_mask : ants.ANTsImage
        Intracranial volume mask
    brain_mask : ants.ANTsImage
        Brain mask from skull stripping
    debug_mode : bool
        Whether to output debug information
        
    Returns
    -------
    tuple
        (icv_mask, is_valid) - mask and validation status
    """
    icv_data = icv_mask.numpy()
    brain_data = brain_mask.numpy()
    
    # Calculate how much brain tissue is outside ICV
    brain_outside_icv = (brain_data > 0) & (icv_data == 0)
    n_outside = np.sum(brain_outside_icv)
    brain_volume = np.sum(brain_data > 0)
    
    if brain_volume == 0:
        log_error("Brain mask is empty - cannot validate ICV")
        return icv_mask, False
    
    outside_ratio = n_outside / brain_volume
    
    # Evaluate severity of misalignment
    if outside_ratio > 0.02:  # More than 2% outside
        log_error(f"{outside_ratio*100:.1f}% of brain mask is outside ICV - ICV segmentation likely failed")
        
        if debug_mode:
            # Check if these are large connected regions (indicating major dents)
            labeled_outside = skm.label(brain_outside_icv)
            if labeled_outside.max() > 0:
                cluster_sizes = [np.sum(labeled_outside == i) for i in range(1, labeled_outside.max() + 1)]
                max_cluster = max(cluster_sizes)
                log_debug(f"Largest brain cluster outside ICV: {max_cluster} voxels")
        
        return icv_mask, False
        
    elif outside_ratio > 0.005:  # 0.5-2% outside
        log_warning(f"{outside_ratio*100:.1f}% of brain outside ICV - minor edge misalignment")
        return icv_mask, True
        
    else:
        log_success(f"ICV mask validated - only {outside_ratio*100:.3f}% of brain outside ICV")
        return icv_mask, True

#=============================================
# Main processing function
#=============================================
def main(argv):
    """
    Main pipeline for brain extraction and tissue segmentation.
    
    The pipeline is modular - components are executed based on requested outputs:
    - Core: bias correction + skull stripping (always runs)
    - Optional: intensity normalization, GM/WM segmentation, ICV extraction
    
    Parameters
    ----------
    argv : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    int
        0 for success, 1 for failure
    """
    # Initialize variables
    infile = argv.i
    outfile = argv.o
    debug_mode = argv.debug
    stats = []
    temp_files = []  # Track temporary files for cleanup
    
    # Set up subject ID for statistics
    if argv.sub is None:
        argv.sub = os.path.basename(infile)
    stats.append(["ID", "%s" % argv.sub])
    
    log_info(f"Processing: {argv.sub}")
    log_debug(f"Debug mode: {'ON' if debug_mode else 'OFF'}", debug_mode)
    
    #############
    # Validation checks
    #############
    # Check input file exists
    if not os.path.isfile(infile):
        log_error(f"Input file not found: {infile}")
        return 1
    
    # Check external tools
    if not check_external_tools():
        log_error("Required external tools not found. Please install FreeSurfer.")
        return 1
    
    # Load input image
    try:
        mi = ants.image_read(infile)
        voxel_vol = np.prod(mi.spacing)
        log_success(f"Loaded image: {mi.shape}, spacing: {mi.spacing}")
    except Exception as e:
        log_error(f"Failed to load input image: {e}")
        return 1
    
    # Check for negative intensities
    if np.any(mi.numpy() < 0):
        log_warning("Negative intensity values detected - adjusting by adding minimum value")
        mi = mi.new_image_like(mi.numpy() + np.abs(np.min(mi.numpy())))
    
    #############
    # Setup output directory
    #############
    outdir = os.path.dirname(outfile) if outfile else os.path.dirname(argv.brain) if argv.brain else '.'
    if outdir == '':
        outdir = '.'
    
    if not os.path.isdir(outdir):
        try:
            os.makedirs(outdir)
            log_info(f"Created output directory: {outdir}")
        except Exception as e:
            log_error(f"Failed to create output directory: {e}")
            return 1
    
    #############
    # STEP 1: Initial bias field correction
    #############
    log_info("Step 1: Initial bias field correction")
    try:
        mi = ants.n4_bias_field_correction(mi)
        log_success("Bias field correction completed")
    except Exception as e:
        log_error(f"Bias field correction failed: {e}")
        return 1
    
    # Save initial bias corrected image for skull stripping
    temp_bias_file = os.path.join(outdir, f"temp_bias_{uuid.uuid4()}.nii.gz")
    temp_files.append(temp_bias_file)
    mi.to_filename(temp_bias_file)
    
    #############
    # STEP 2: Skull stripping with SynthStrip
    #############
    log_info("Step 2: Brain extraction using SynthStrip")
    
    # Determine brain mask file
    if argv.brain is not None:
        brainfile = argv.brain
    else:
        brainfile = os.path.join(outdir, f"temp_brain_{uuid.uuid4()}.nii.gz")
        temp_files.append(brainfile)
    
    # Run SynthStrip if mask doesn't exist
    if not os.path.isfile(brainfile):
        synthstrip_cmd = ["mri_synthstrip", "-i", temp_bias_file, "-m", brainfile, "-b", "0"]
        
        if argv.gpu:
            synthstrip_cmd.append("-g")
            log_info("Using GPU acceleration for skull stripping")
        
        log_debug(f"Running command: {' '.join(synthstrip_cmd)}", debug_mode)
        
        try:
            call(synthstrip_cmd)
            log_success("Brain extraction completed")
        except CalledProcessError as e:
            log_error(f"SynthStrip failed: {e}")
            return 1
    else:
        log_info(f"Using existing brain mask: {brainfile}")
    
    # Load brain mask
    try:
        mi_mask = ants.image_read(brainfile)
        brain_vol = voxel_vol * np.sum(mi_mask.numpy() > 0)
        stats.append(["Brain_volume", "%f" % brain_vol])
        log_success(f"Brain volume: {brain_vol:.2f} mm³")
    except Exception as e:
        log_error(f"Failed to load brain mask: {e}")
        return 1
    
    # Validate brain mask
    if np.sum(mi_mask.numpy() > 0) == 0:
        log_error("Brain mask is empty - skull stripping failed")
        return 1
    
    # Fix potential header mismatches
    ants.core.ants_image.copy_image_info(mi, mi_mask)
    
    #############
    # STEP 3: Second bias correction (if needed)
    #############
    need_second_bias = argv.bias or argv.norm or argv.o
    
    if need_second_bias:
        log_info("Step 3: Second bias field correction with brain mask")
        try:
            mi = ants.n4_bias_field_correction(mi, mi_mask)
            log_success("Masked bias field correction completed")
        except Exception as e:
            log_error(f"Second bias correction failed: {e}")
            return 1
        
        # Save bias corrected image if requested
        if argv.bias:
            mi.to_filename(argv.bias)
            log_success(f"Saved bias-corrected image: {argv.bias}")
    else:
        log_info("Step 3: Skipping second bias correction (not needed)")
    
    #############
    # STEP 4: Intensity normalization (if needed)
    #############
    need_normalization = argv.norm or argv.o
    
    if need_normalization:
        log_info("Step 4: Intensity normalization")
        try:
            img, intnorm = rescale(mi.numpy(), mask=mi_mask.numpy())
            mi_norm = mi.new_image_like(img)
            stats.append(["NAWM_intensity", "%f" % intnorm])
            log_success(f"Intensity normalized (WM peak: {intnorm:.2f})")
            
            # Save normalized image if requested
            if argv.norm:
                mi_norm.to_filename(argv.norm)
                log_success(f"Saved normalized image: {argv.norm}")
                
        except Exception as e:
            log_error(f"Intensity normalization failed: {e}")
            return 1
    else:
        log_info("Step 4: Skipping intensity normalization (not needed)")
        mi_norm = None
    
    #############
    # STEP 5: GM/WM segmentation (if output requested)
    #############
    if argv.o:
        log_info("Step 5: Grey/white matter segmentation")
        
        if mi_norm is None:
            log_error("GM/WM segmentation requires intensity normalization")
            return 1
        
        try:
            # Simple threshold-based approach for CSF-dark sequences (T1, FLAIR)
            # TODO: For T2/non-CSF-dark sequences, use SynthSeg tissue masks
            updated_mask = np.multiply(
                sn.gaussian_filter(mi_norm.numpy(), sigma=(0.75, 0.75, 0), order=0) > 0.375,
                mi_mask.numpy()
            )
            
            gmwm_vol = voxel_vol * np.sum(updated_mask)
            stats.append(["GMWM_volume", "%f" % gmwm_vol])
            log_success(f"GM/WM volume: {gmwm_vol:.2f} mm³")
            
        except Exception as e:
            log_error(f"GM/WM segmentation failed: {e}")
            return 1
    else:
        log_info("Step 5: Skipping GM/WM segmentation (output not requested)")
        updated_mask = None
    
    #############
    # STEP 6: ICV extraction (optional)
    #############
    if argv.icv is not None:
        log_info("Step 6: Intracranial volume extraction using SynthSeg")
        
        icvfile = argv.icv
        
        if not os.path.isfile(icvfile):
            # Build SynthSeg command
            synthseg_cmd = ["mri_synthseg", "--i", temp_bias_file, "--robust", 
                          "--keepgeom", "--o", icvfile]
            
            # Add parcellation if requested
            if argv.parc:
                synthseg_cmd.append("--parc")
                log_info("Parcellation output enabled")
            
            # GPU or CPU mode
            if argv.gpu:
                synthseg_cmd.append("--gpu")
                log_info("Using GPU acceleration for SynthSeg")
            else:
                n_threads = get_optimal_threads()
                synthseg_cmd.extend(["--cpu", "--threads", str(n_threads)])
                log_info(f"Using CPU with {n_threads} threads")
            
            log_debug(f"Running command: {' '.join(synthseg_cmd)}", debug_mode)
            
            try:
                call(synthseg_cmd)
                log_success("SynthSeg segmentation completed")
            except CalledProcessError as e:
                log_error(f"SynthSeg failed: {e}")
                return 1
            
            # Load and binarize segmentation for ICV mask
            try:
                segfile = ants.image_read(icvfile)
                icv_mask = ants.utils.threshold_image(segfile, 1e-15)
                icv_mask.to_filename(icvfile)
            except Exception as e:
                log_error(f"Failed to process ICV mask: {e}")
                return 1
        else:
            log_info(f"Using existing ICV mask: {icvfile}")
            icv_mask = ants.image_read(icvfile)
        
        # Validate ICV mask
        icv_mask, is_valid = validate_icv_mask(icv_mask, mi_mask, debug_mode)
        
        if not is_valid:
            log_warning("ICV validation failed - results may be unreliable")
        
        # Calculate ICV volume
        icv_vol = voxel_vol * np.sum(icv_mask.numpy() > 0)
        stats.append(["ICV_volume", "%f" % icv_vol])
        log_success(f"ICV volume: {icv_vol:.2f} mm³")
        
        # Constrain GM/WM mask to ICV if both exist
        if updated_mask is not None:
            gmwm_outside = np.sum((updated_mask > 0) & (icv_mask.numpy() == 0))
            if gmwm_outside > 0:
                log_warning(f"{gmwm_outside} GM/WM voxels outside ICV - constraining to ICV")
                updated_mask = np.multiply(updated_mask, icv_mask.numpy())
    else:
        log_info("Step 6: Skipping ICV extraction (not requested)")
    
    #############
    # STEP 7: Save final GM/WM mask
    #############
    if argv.o and updated_mask is not None:
        log_info("Step 7: Saving final GM/WM mask")
        try:
            out = mi_mask.new_image_like(updated_mask)
            out.to_filename(outfile)
            log_success(f"Saved GM/WM mask: {outfile}")
        except Exception as e:
            log_error(f"Failed to save output mask: {e}")
            return 1
    
    #############
    # STEP 8: Save statistics
    #############
    if argv.stats is not None:
        log_info("Step 8: Saving statistics")
        try:
            import csv
            stats = np.array(stats).T.tolist()
            with open(argv.stats, 'w', newline='') as fid:
                writer = csv.writer(fid)
                writer.writerows(stats)
            log_success(f"Saved statistics: {argv.stats}")
        except Exception as e:
            log_error(f"Failed to save statistics: {e}")
    
    #############
    # Cleanup temporary files
    #############
    if not debug_mode:
        log_info("Cleaning up temporary files")
        for temp_file in temp_files:
            if os.path.isfile(temp_file):
                try:
                    os.remove(temp_file)
                    log_debug(f"Removed: {temp_file}", debug_mode)
                except:
                    pass
    else:
        log_debug("Debug mode: keeping temporary files", debug_mode)
        for temp_file in temp_files:
            if os.path.isfile(temp_file):
                log_debug(f"Kept temporary file: {temp_file}", debug_mode)
    
    log_success("Pipeline completed successfully!")
    return 0

#=============================================
# Command-line interface
#=============================================
if __name__ == "__main__":
    try:
        # Set up command-line parser
        parser = OptionParser(
            description='MR-CLOVER: Modular brain extraction and tissue segmentation pipeline for clinical MRI.',
            epilog='Example: python mr_clover.py -i T1.nii.gz -o gmwm_mask.nii.gz --brain brain.nii.gz --icv icv.nii.gz'
        )
        
        # Required arguments
        parser.add_option('-i', dest='i', 
                         help='Input NIFTI image (required)', 
                         metavar='FILE')
        
        # Optional outputs (pipeline runs modules based on these)
        parser.add_option('-o', dest='o', 
                         help='Output grey/white matter mask (triggers intensity norm + GM/WM segmentation)', 
                         metavar='FILE', default=None)
        parser.add_option('--brain', dest='brain', 
                         help='Output brain mask from skull stripping', 
                         metavar='FILE', default=None)
        parser.add_option('--icv', dest='icv', 
                         help='Output intracranial volume mask (triggers SynthSeg)', 
                         metavar='FILE', default=None)
        parser.add_option('--norm', dest='norm', 
                         help='Output intensity-normalized brain image', 
                         metavar='FILE', default=None)
        parser.add_option('--bias', dest='bias', 
                         help='Output bias field corrected image', 
                         metavar='FILE', default=None)
        
        # Additional options
        parser.add_option('--stats', dest='stats', 
                         help='Output CSV file with volumes and normalization values', 
                         metavar='FILE', default=None)
        parser.add_option('--sub', dest='sub', 
                         help='Subject ID for statistics file (defaults to input filename)', 
                         metavar='STRING', default=None)
        
        # Processing options
        parser.add_option('--gpu', dest='gpu', 
                         help='Enable GPU acceleration (requires NVIDIA GPU with CUDA)', 
                         default=False, action="store_true")
        parser.add_option('--parc', dest='parc', 
                         help='Enable parcellation output from SynthSeg', 
                         default=False, action="store_true")
        parser.add_option('--debug', dest='debug', 
                         help='Enable debug mode (keeps temporary files, verbose output)', 
                         default=False, action="store_true")
        
        (options, args) = parser.parse_args()
        
        # Validate required arguments
        if not options.i:
            parser.error("Input file (-i) is required")
        
        # Check if at least one output is specified
        if not any([options.o, options.brain, options.icv, options.norm, options.bias, options.stats]):
            parser.error("At least one output must be specified")
        
        # Run main pipeline
        sys.exit(main(options))
        
    except KeyboardInterrupt:
        log_error("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        sys.exit(1)