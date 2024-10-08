
# MR-CLOVER: MRI CLinical resOlution brain VolumEtRics

## Description

MR-CLOVER (Tool for Assessing MR-based Clinical resOlution brain VolumEtRics) is a Python-based pipeline that combines publicly available tools from ANTs and Freesurfer to extract the brain from clinical MRI scans and return segmented grey and white matter masks. It facilitates various MRI image processing tasks, including bias field correction, brain extraction, intensity normalization, and intracranial volume segmentation using advanced computational methods.

This is what the pipeline looks like:

![mr-clover pipeline](pipeline.jpg)

## Features

- **Bias Field Correction**: Adjusts the MRI scan to minimize intensity inhomogeneities.
- **Brain Extraction**: Segments the brain from the surrounding skull and other non-brain structures.
- **Intensity Normalization**: Standardizes the intensity values across different scans for consistent analysis.
- **Grey/White Matter Segmentation**: Segments and masks grey and white matter within the extracted brain.
- **Intracranial Volume Calculation**: Optionally, calculates the total intracranial volume (ICV).
  
## Installation

### Requirements
- Python 3.x
- NumPy
- SciPy
- scikit-image
- scikit-learn
- ANTs (Advanced Normalization Tools)
- Freesurfer
- Warnings module for handling potential errors during execution.

Ensure that all dependencies are installed using pip:

```bash
pip install -r requirements.txt
```

ANTs and Freesurfer need to be installed separately. Please see the corresponding websites for instructions.

### Setup

Clone the repository or download the source code from GitHub. No additional setup is required beyond ensuring the required libraries are installed.

## Usage

To use MR-CLOVER, prepare your MRI data in NIfTI format and run the tool from the command line. Here is a basic example:

```bash
python mrclover.py -i path/to/input_image.nii.gz -o path/to/output_mask.nii.gz
```

### Command Line Arguments

- `-i`: Input NIfTI image file.
- `-o`: Output file for the grey/white-matter mask.
- `--brain`: Output file for the brain mask.
- `--icv`: Output file for the intracranial volume mask.
- `--norm`: Output file for the intensity normalized brain image.
- `--bias`: Output file for the bias field corrected image.
- `--stats`: Output file to save volumes and intensity normalization values.
- `--sub`: Subject ID to be saved in the stats file. If none is given, the input file name is used.

### Example

```bash
python mrclover.py -i patient.nii.gz -o gmwm_mask.nii.gz --brain brain_mask.nii.gz --icv icv_mask.nii.gz --norm normalized.nii.gz --bias bias_corrected.nii.gz --stats volumes.csv
```

## Development

MR-CLOVER is an open-source project developed by MDS at Massachusetts General Hospital and Harvard Medical School. Contributions are welcome, particularly in improving the algorithms for brain tissue segmentation and volume calculation.

## Contact

For issues, suggestions, or contributions, please contact the lead developer.

## Acknowledgements
If you find this tool useful for your work, please cite 

[1] Alhadid, Kenda, et al. "Brain volume is a better biomarker of outcomes in ischemic stroke compared to brain atrophy." arXiv preprint arXiv:2403.12788 (2024).

as well as the relevant citations for ANTs and the Freesurfer packages Synthstrip and Synthseg.

## Version History

- **0.2** (March 2024): Latest release with enhanced feature set and performance improvements compared to the tools found in WMHP.

Feel free to reach out or contribute to making MR-CLOVER a more robust tool for clinical and research applications in neuroimaging!
