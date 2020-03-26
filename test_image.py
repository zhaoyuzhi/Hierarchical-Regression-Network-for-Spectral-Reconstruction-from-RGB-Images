import torch
import scipy.io as io
import numpy as np
import cv2

img = cv2.imread('/home/SENSETIME/zhaoyuzhi/Documents/NTIRE 2020 spectral reconstruction/NTIRE2020_Train_Clean/ARAD_HS_0001_clean.png', -1)
print(img.shape)

cv2.imshow('1', img)
cv2.waitKey(5000)
