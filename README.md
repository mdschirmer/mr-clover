# MR-CLOVER: MRI CLinical resOlution brain VolumEtRics

## Description

MR-CLOVER (Tool for Assessing MR-based Clinical resOlution brain VolumEtRics) is a Python-based pipeline that combines publicly available tools from ANTs and Freesurfer to extract the brain from clinical MRI scans and return segmented grey and white matter masks. SynthSeg is the core segmentation engine, so it works with any MRI contrast.

Pipeline steps:

1. Bias field correction, pass 1 (ANTs N4, unmasked)
2. SynthSeg segmentation (FreeSurfer)
3. Estimate the white matter intensity mode
4. Build a normal-appearing white matter mask (excludes pathology)
5. Bias field correction, pass 2 (ANTs N4, masked to grey/white matter)
6. Combine the two bias fields
7. Derive all outputs from the SynthSeg labels: brain mask, GM/WM mask, ICV mask, ventricle mask, and optionally an intensity-normalized image and the combined bias field

This is what the pipeline looks like:

![mr-clover pipeline](pipeline.jpg)

## Features

- **Any MRI contrast**: All masks come from SynthSeg labels, not from contrast-specific thresholds
- **Two-pass bias field correction**: The second N4 pass is restricted to grey/white matter; the combined bias field can be saved for on-the-fly correction
- **Lesion-robust intensities**: Tissue intensity distributions come from normal-appearing tissue, so lesions do not skew them
- **Intensity Normalization**: Scales the white matter mode to a fixed target (default 0.75)
- **Masks**: Brain, grey/white matter, intracranial volume (ICV) and ventricles
- **Ventricle refinement**: Probabilistic Gaussian classification, hole filling and connectivity constraint; dilation is in-plane only
- **Statistics**: Volumes and the white matter mode intensity in a CSV file
- **Debug Mode**: Keep intermediate files and get verbose output for troubleshooting
- **GPU Acceleration**: SynthSeg uses the GPU unless `--cpu` is given

## What's New in Version 0.5.0.0

This version replaces the SynthStrip-based pipeline of 0.4 with the SynthSeg-based rewrite that was first developed inside WMHP (as 0.3.0).

- SynthSeg replaces SynthStrip for brain extraction; all masks are derived from SynthSeg labels
- Two-pass N4 bias field correction; the second pass uses the GM/WM mask (more robust for high-resolution images)
- New outputs: `--bias-field` (combined bias field) and `--synthseg` (SynthSeg segmentation, also reused as input)
- Statistics now include `wm_mode_intensity`; volume columns are named `*_volume_mm3`
- Options are now given with full output paths; `--sub` is now `--subject`, `--parc` was removed
- Fixed ANTsPy N4 crash on large images; faster mode estimation on high-resolution images
- Versions now use four numbers: MAJOR.FEATURE.TESTED.ITERATION

## Installation

### Requirements
- Python 3.x
- NumPy
- SciPy
- ANTsPy (`antspyx`)
- Freesurfer 7.3 or newer (specifically mri_synthseg)

Ensure that all dependencies are installed using pip:

```bash
pip install -r requirements.txt
```

ANTs and Freesurfer need to be installed separately. Please see the corresponding websites for instructions.

### Setup

Clone the repository or download the source code from GitHub. No additional setup is required beyond ensuring the required libraries are installed.

## Usage

To use MR-CLOVER, prepare your MRI data in NIfTI format and run the tool from the command line. Give a file path for each output you want; at least one output is required.

### Basic Examples

**GM/WM mask:**
```bash
python mr_clover.py -i patient.nii.gz -o gmwm_mask.nii.gz
```

**Full pipeline with bias field output:**
```bash
python mr_clover.py -i patient.nii.gz -o gmwm_mask.nii.gz --brain brain_mask.nii.gz \
    --icv icv_mask.nii.gz --bias-field bias_field.nii.gz --stats volumes.csv
```

**Custom normalization target:**
```bash
python mr_clover.py -i patient.nii.gz --norm normalized.nii.gz --norm-target 1.0
```

### Command Line Arguments

#### Required
- `-i`, `--input`: Input NIfTI image file

#### Outputs (at least one required)
- `-o`, `--output`: Grey/white matter mask
- `--brain`: Brain mask
- `--icv`: Intracranial volume mask
- `--ventricles`: Ventricle mask
- `--synthseg`: SynthSeg segmentation (if the file exists, it is used as input instead of running SynthSeg again)
- `--norm`: Intensity-normalized image
- `--bias`: Bias-corrected image
- `--bias-field`: Combined bias field (corrected image = input / bias field)
- `--stats`: CSV file with volumes (`*_volume_mm3`) and `wm_mode_intensity`

#### Additional Options
- `-s`, `--subject`: Subject ID for the statistics file
- `--norm-target`: Target white matter intensity for normalization (default: 0.75)
- `--no-refine`: Skip intensity refinement, use raw SynthSeg labels
- `--cpu`: Run SynthSeg on the CPU instead of the GPU
- `--robust`: Use SynthSeg robust mode (slower)
- `-f`, `--force`: Process again even if outputs exist
- `-v`, `--verbose`: Verbose output
- `--debug`: Debug mode (keeps temporary files, verbose output)
- `--version`: Show the version

## Known Limitations

- SynthSeg always runs, even if only one output is requested
- The pipeline image above shows the 0.4 pipeline (SynthStrip-based)

## Development

MR-CLOVER is an open-source project developed by MDS at Massachusetts General Hospital and Harvard Medical School. Contributions are welcome, particularly in improving the algorithms for brain tissue segmentation and volume calculation.

## Contact

For issues, suggestions, or contributions, please contact the lead developer at mschirmer1@mgh.harvard.edu.

## Acknowledgements
If you find this tool useful for your work, please cite:

[1] Alhadid, Kenda, et al. "Brain volume is a better biomarker of outcomes in ischemic stroke compared to brain atrophy." arXiv preprint arXiv:2403.12788 (2024).

Please also cite the relevant papers for ANTs and the Freesurfer packages SynthStrip and SynthSeg.

## Version History

- **0.5.0.0** (November 2026): SynthSeg-based rewrite for any MRI contrast, two-pass bias field correction, lesion-robust intensities, new output options
- **0.4.1** (April 2026): SynthStrip-based pipeline with ventricle mask output; stops if SynthStrip/SynthSeg fail
- **0.3** (January 2025): Major refactoring with modular architecture, enhanced error handling, ICV validation, and debug mode
- **0.2** (March 2024): Latest stable release with enhanced feature set and performance improvements compared to the tools found in WMHP

Feel free to reach out or contribute to making MR-CLOVER a more robust tool for clinical and research applications in neuroimaging!