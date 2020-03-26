import os
import torch
import scipy.io as io
import numpy as np
import cv2

import utils

sample_folder = os.path.join('train_visualization')
utils.check_path(sample_folder)

spectral = io.loadmat('./NTIRE2020_Train_Spectral/ARAD_HS_0031.mat')
#print(spectral)            # 'cube' is the data; 'band' is the spectral; 'norm_factor'
img = spectral['cube']      # 31 channels (482,512,31)
print(img.shape)
print(img.dtype)
img = (img * 255).astype(np.uint8)

temp = img[:,:,0]
save_img_name = '0' + '.png'
save_img_path = os.path.join(sample_folder, save_img_name)
cv2.imwrite(save_img_path, temp)
temp = img[:,:,1]
save_img_name = '1' + '.png'
save_img_path = os.path.join(sample_folder, save_img_name)
cv2.imwrite(save_img_path, temp)
temp = img[:,:,2]
save_img_name = '2' + '.png'
save_img_path = os.path.join(sample_folder, save_img_name)
cv2.imwrite(save_img_path, temp)
temp = img[:,:,10]
save_img_name = '10' + '.png'
save_img_path = os.path.join(sample_folder, save_img_name)
cv2.imwrite(save_img_path, temp)
temp = img[:,:,20]
save_img_name = '20' + '.png'
save_img_path = os.path.join(sample_folder, save_img_name)
cv2.imwrite(save_img_path, temp)
temp = img[:,:,30]
save_img_name = '30' + '.png'
save_img_path = os.path.join(sample_folder, save_img_name)
cv2.imwrite(save_img_path, temp)
