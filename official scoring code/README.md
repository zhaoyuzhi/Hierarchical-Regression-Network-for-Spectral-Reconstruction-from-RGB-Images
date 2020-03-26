# NTIRE2020 Challenge on Spectral Reconstruction from RGB Images

This repository contains supporting code for the [NTIRE 2020](http://www.vision.ee.ethz.ch/ntire20/) Spectral Reconstruction challenge held in conjunction with [CVPR 2020](http://cvpr2020.thecvf.com/program/workshops).


The challenge includes two tracks:
* [Track 1: “Clean”](https://competitions.codalab.org/competitions/22225) recovering hyperspectral data from uncompressed 8-bit RGB images created by applying a known response function to ground truth hyperspectral information.


* [Track 2: “Real World”](https://competitions.codalab.org/competitions/22226) recovering hyperspectral data from jpg-compressed 8-bit RGB images created by applying an unknown response function to ground truth hyperspectral information.

## Data Access
450 hyperspectral training images and their corresponding "Clean" and "Real World" images are available on the challenge track websites above, registration is required to access data.

## Example code
The `clean_example.ipynb` and `real_world_example.ipynb` Jupyter notebooks include example code demonstrating how "Clean" and "Real World" training images were created.

## Libraries
`SpectralUtils.py` includes utilities for handling spectral images and projecting them to RGB.
`EvalMetrics.py` includes code used to measure reconstruction accuracy - there are the metrics which shall be used to score participants during the challehnge.

## Resources
The resource directory contains:
* `cie_1964_w_gain` - the response function used in the "Clean" track.
* `example_D40_camera_w_gain` - an _example_ physical camera response function, somewhat similar to that used in the "Real World" track.
* `sample_hs_img_001.mat` - a sample spectral image, additional images are available on the challenge track websites above.

## :warning: Notice
While `SpectralUtils` contains some _example_ noise parameters, and an _example_ camera response function is included in the `resources` folder, __the "Real World" track will be using *different* noise parameters and a *different* camera response function which will remain confidential throughout the challenge__.
