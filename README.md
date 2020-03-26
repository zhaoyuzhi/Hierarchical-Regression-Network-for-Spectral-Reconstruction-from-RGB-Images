The README file for NTIRE 2020 Spectral Reconstruction Challenge of Team OrangeCat: Hierarchical Regression Network for Spectral Reconstruction from RGB Images

# File structure

```
NTIRE 2020 Spectral Reconstruction Challenge
│   README.md
│   validation*.py
│   test*.py
│   ensemble*.py
│
└───track1 (saving the trained models of track1)
│   │   code1_G_epoch9000_bs8.pth
│   │   code1_second_G_epoch8000_bs8.pth
│   │   ...
│
└───track2 (saving the trained models of track2)
│   │   code1_bs2_G_epoch6000_bs2.pth
│   │   code2_G_epoch6000_bs8.pth
│   │   ...
|
└───NTIRE2020_Test_Clean
│    │   ARAD_HS_0468_clean.mat
│    │   ARAD_HS_0508_clean.mat
│    │   ...
│
└───NTIRE2020_Test_RealWorld
│    │   ARAD_HS_0477_RealWorld.mat
│    │   ARAD_HS_0502_RealWorld.mat
│    │   ...
│
└───test (will generate by test1.py or test2.py)
│   └───track1
│       │   ARAD_HS_0468_clean.mat
│       │   ARAD_HS_0508_clean.mat
│       │   ...
│   └───track2
│       │   ARAD_HS_0477_RealWorld.mat
│       │   ARAD_HS_0502_RealWorld.mat
│       │   ...
│
└───ensemble (will generate by ensemble1.py or ensemble2.py)
│   └───track1
│       │   ARAD_HS_0468_clean.mat
│       │   ARAD_HS_0508_clean.mat
│       │   ...
│   └───track2
│       │   ARAD_HS_0477_RealWorld.mat
│       │   ARAD_HS_0502_RealWorld.mat
│       │   ...
│   
```

# Requirements
* Python 3.6
* Pytorch 1.0.0
* Cuda 8.0

# Train
* Run `train.py`.
* Change `baseroot` that contains training data.
* Change `save_path` corresponding to track 1 or track 2.
* Change other parameters.

# Test
## track 1 generation
* Run `test1.py`.
* It will output 8 results of 8 networks.
## track 1 ensemble
* Run `ensemble_track1_8methods.py`.
* It will output 1 ensemble result of 8 generated data.
## track 2 generation
* Run `test2.py`.
* It will output 8 results of 8 networks.
## track 2 ensemble
* Run `ensemble_track2_8methods.py`.
* It will output 1 ensemble result of 8 generated data.

# Visualize
* Run `train_visualize.py` or `validation_visualize.py`.
