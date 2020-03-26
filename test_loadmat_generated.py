import torch
import scipy.io as io
import numpy as np
import cv2
import hdf5storage as hdf5

import utils

spectral = hdf5.loadmat('./ensemble/track1/ARAD_HS_0466.mat')
choice = 3
print(spectral)             # 'cube' is the data; 'band' is the spectral; 'norm_factor'
img = spectral['cube']      # 31 channels (482,512,31)
print(img.shape)
print(img.dtype)

if choice == 1:
    a = 'ARAD_HS_0001.mat'
    print(a[:12])
if choice == 2:
    img = spectral['cube'][:,:,10]
    a = (img * 255).astype(np.uint8)
    cv2.imshow('1', a)
    cv2.waitKey(0)
if choice == 3:
    b = cv2.resize(img, (256, 256))
    print(b.shape)
    cv2.imshow('1', b[:,:,20])
    cv2.waitKey(0)
