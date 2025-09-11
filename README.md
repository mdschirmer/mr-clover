# MR-CLOVER: MRI CLinical resOlution brain VolumEtRics

## Description

MR-CLOVER (Tool for Assessing MR-based Clinical resOlution brain VolumEtRics) is a Python-based pipeline that combines publicly available tools from ANTs and Freesurfer to extract the brain from clinical MRI scans and return segmented grey and white matter masks. It facilitates various MRI image processing tasks, including bias field correction, brain extraction, intensity normalization, and intracranial volume segmentation using advanced computational methods.

The pipeline is **modular** - each processing step is only executed when its corresponding output is requested, allowing for flexible usage from simple skull stripping to comprehensive tissue segmentation.

This is what the pipeline looks like:

![mr-clover pipeline](pipeline.jpg)

## Features

- **Modular Pipeline**: Run only the components you need - each output triggers its required processing steps
- **Bias Field Correction**: Adjusts the MRI scan to minimize intensity inhomogeneities (dual-pass correction available)
- **Brain Extraction**: Segments the brain from the surrounding skull and other non-brain structures using SynthStrip
- **Intensity Normalization**: Standardizes the intensity values across different scans for consistent analysis
- **Grey/White Matter Segmentation**: Segments and masks grey and white matter within the extracted brain
- **Intracranial Volume Calculation**: Optionally calculates the total intracranial volume (ICV) using SynthSeg
- **Quality Validation**: Automatic validation of ICV masks to detect segmentation failures
- **Color-coded Logging**: Clear visual feedback with color-coded error, warning, success, and debug messages
- **Debug Mode**: Keep intermediate files and get verbose output for troubleshooting
- **Smart Resource Management**: Automatically detects and uses optimal number of CPU threads
- **GPU Acceleration**: Optional GPU support for SynthStrip and SynthSeg (requires NVIDIA GPU with CUDA)

## What's New in Version 0.3

- **Modular Architecture**: Pipeline components now run only when their outputs are requested
- **Enhanced Error Handling**: Comprehensive validation checks with informative color-coded messages
- **ICV Validation**: Basic, automatic detection of ICV segmentation failures
- **Improved SynthSeg Integration**: Uses `--keepgeom` flag to eliminate registration step, `--robust` for better quality
- **Smart Threading**: Automatically uses 90% of available CPU cores for optimal performance
- **Debug Mode**: New `--debug` flag for keeping temporary files and verbose output
- **Optional Parcellation**: Added `--parc` flag for SynthSeg parcellation output

## Installation

### Requirements
- Python 3.x
- NumPy
- SciPy
- scikit-image
- ANTs (Advanced Normalization Tools)
- Freesurfer (specifically mri_synthstrip and mri_synthseg)
- Warnings module for handling potential errors during execution

Ensure that all dependencies are installed using pip:

```bash
pip install -r requirements.txt
```

ANTs and Freesurfer need to be installed separately. Please see the corresponding websites for instructions.

### Setup

Clone the repository or download the source code from GitHub. No additional setup is required beyond ensuring the required libraries are installed.

## Usage

To use MR-CLOVER, prepare your MRI data in NIfTI format and run the tool from the command line. The pipeline is modular - it will only run the processing steps needed for your requested outputs.

### Basic Examples

**Simple brain extraction only:**
```bash
python mr_clover.py -i patient.nii.gz --brain brain_mask.nii.gz
```

**Bias correction only:**
```bash
python mr_clover.py -i patient.nii.gz --bias bias_corrected.nii.gz
```

**Full pipeline with GM/WM segmentation:**
```bash
python mr_clover.py -i patient.nii.gz -o gmwm_mask.nii.gz
```

**Complete processing with all outputs:**
```bash
python mr_clover.py -i patient.nii.gz -o gmwm_mask.nii.gz --brain brain_mask.nii.gz --icv icv_mask.nii.gz --norm normalized.nii.gz --bias bias_corrected.nii.gz --stats volumes.csv
```

### Command Line Arguments

#### Required
- `-i`: Input NIfTI image file

#### Optional Outputs (at least one required)
- `-o`: Output grey/white matter mask (triggers intensity normalization + GM/WM segmentation)
- `--brain`: Output brain mask from skull stripping
- `--icv`: Output intracranial volume mask (triggers SynthSeg)
- `--norm`: Output intensity-normalized brain image
- `--bias`: Output bias field corrected image
- `--stats`: Output CSV file with volumes and normalization values

#### Additional Options
- `--sub`: Subject ID for statistics file (defaults to input filename)
- `--gpu`: Enable GPU acceleration (requires NVIDIA GPU with CUDA)
- `--parc`: Enable parcellation output from SynthSeg
- `--debug`: Enable debug mode (keeps temporary files, verbose output)

### Pipeline Behavior

The pipeline intelligently determines which processing steps to run based on your requested outputs:

1. **Core steps** (always run): Initial bias correction → Brain extraction
2. **Second bias correction**: Only if `--bias`, `--norm`, or `-o` specified
3. **Intensity normalization**: Only if `--norm` or `-o` specified
4. **GM/WM segmentation**: Only if `-o` specified
5. **ICV extraction**: Only if `--icv` specified

## Known Limitations

- Currently optimized for CSF-dark modalities (T1, FLAIR)
- GM/WM segmentation uses simple thresholding (GMM-based refinement planned for v0.4)
- Future versions will support T2 and other modalities using SynthSeg tissue masks for normalization

## Development

MR-CLOVER is an open-source project developed by MDS at Massachusetts General Hospital and Harvard Medical School. Contributions are welcome, particularly in improving the algorithms for brain tissue segmentation and volume calculation.

### Planned Improvements (v0.4)
- Support for non-CSF-dark modalities (T2, etc.)
- Use SynthSeg tissue masks for robust intensity normalization
- GMM-based tissue segmentation with seed points from SynthSeg labels
- Enhanced tissue classification for all MRI sequences

## Contact

For issues, suggestions, or contributions, please contact the lead developer at mschirmer1@mgh.harvard.edu.

## Acknowledgements
If you find this tool useful for your work, please cite:

[1] Alhadid, Kenda, et al. "Brain volume is a better biomarker of outcomes in ischemic stroke compared to brain atrophy." arXiv preprint arXiv:2403.12788 (2024).

Please also cite the relevant papers for ANTs and the Freesurfer packages SynthStrip and SynthSeg.

## Version History

- **0.3** (January 2025): Major refactoring with modular architecture, enhanced error handling, ICV validation, and debug mode
- **0.2** (March 2024): Latest stable release with enhanced feature set and performance improvements compared to the tools found in WMHP

Feel free to reach out or contribute to making MR-CLOVER a more robust tool for clinical and research applications in neuroimaging!