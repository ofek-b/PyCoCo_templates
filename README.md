This is a fork of [this repo](https://github.com/maria-vincenzi/PyCoCo_templates) by [Vincenzi et al. (2019)](https://arxiv.org/abs/1908.05228), with some minor changes.

This code is used in a work in progress.

### Prerequisites
- Python 3 with basic packages
- The Package [george](https://george.readthedocs.io/en/latest/) (version 0.3.1 at least)

### Build your own template: Instructions
1. Set the variables `COCO_PATH` and `SN_NAME` in `config.py`.
1. Automated: Run the function `get_spectroscopy` (uses spectra folder downloaded from [WISeREP](https://wiserep.org/) incl. the csv file) in `run.py`. Or (see example files for formats):
   1. Place a file with the list of the spectra `./Inputs/Spectroscopy/1_spec_lists_original`. Place the actual files with the spectra in `./Inputs/Spectroscopy/1_spec_original`. If you want to smooth the spectra use the provided code. Otherwise, skip this and put list and spectra in the folders `./Inputs/Spectroscopy/2_spec_lists_smoothed` and `./Inputs/Spectroscopy/2_spec_smoothed`.
   1. Modify `./Inputs/SNe_info/info.dat` and add a row for each new template you want to build.
1. Run the function `get_photometry` (downloads photometry from the [OSC](https://sne.space/)) in `run.py`. You will be prompted to input the names (without .dat) of the transmission function files (see next step) corresponding to every photometry entry form OSC. Or (see example files for formats):      
   1. Place your photometry in magnitudes in `./Inputs/Photometry/0_LCs_mags_raw`. If your photometry is already in flux or is already dust corrected (or you don’t want to dust correct it) and/or is already extended at early/late time (or you don’t want to extend it at early/late times) just skip all these steps, do not run the first 4 scripts and place the photometry directly in `./Inputs/Photometry/4_LCs_late_extrapolated`.
1. Put the transmission function files with the name of the band + .dat in `./Inputs/Photometry/filterdict.json`.
1. Write which filters from the photometry file you wish to not use in `./Outputs/SN_NAME/exclude_filt`.
1. Write the names of 4 filters corresponding to V, V, B, B filters in `./Outputs/SN_NAME/VVBB` (for initial strech purposes).
1. Finish filling the line corresponding to the SN in `./Inputs/SNe_info/info.dat`.
1. Run the scripts by order by running the function `runpycoco` in `run.py`.

All the outputs (LC fit, mangled spectra, various plots and final template) will be created in `./Outputs`.

Figure 1 in [Vincenzi et al. (2019)](https://arxiv.org/abs/1908.05228) shows a flow chart of the process where each step of the process corresponds to a script in `./Codes_Scripts`. This fork uses .py scripts instead of .ipynb notebooks.

![Imgur](pycoco_code_structure.png)
