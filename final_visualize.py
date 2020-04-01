import torch
import scipy.io as io
import os
import numpy as np
import cv2
import hdf5storage as hdf5
import matplotlib.pyplot as plt

def split_images(in_path):
    file_list_1 = []
    for filename in os.listdir(in_path):
        split_list = []
        spectral = hdf5.loadmat(in_path + "/" + filename)
        #img = spectral['cube']  # 31 channels (482,512,31)
        #print(img.shape)
        #print(img.dtype)
        split_path = in_path + '_split'
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        for i in range(31):
            img = spectral['cube'][:, :, i]
            a = (img * 255).astype(np.uint8)
            new_dir = filename.replace('.mat', '')
            dir_path_s = split_path + '/' +new_dir
            if not os.path.exists(dir_path_s):
                os.makedirs(dir_path_s)
            new_file = dir_path_s + '/' + new_dir+ '_%d.jpg'%i
            #print(new_file)
            cv2.imwrite(new_file, a)
            #print(a)
            split_list.append(a)
        #print(len(spectral_list))
        file_list_1.append(split_list)
    print(len(file_list_1))
    return file_list_1

def color_images(in_path):
    file_list_2 = []
    for filename in os.listdir(in_path):
        color_list = []
        spectral = hdf5.loadmat(in_path + "\\" + filename)
        color_path = in_path + '_color'
        if not os.path.exists(color_path):
            os.makedirs(color_path)
        for i in range(31):
            img = spectral['cube'][:, :, i]
            a = (img * 255).astype(np.uint8)
            new_dir = filename.replace('.mat', '')
            dir_path_c = color_path + '\\' + new_dir
            if not os.path.exists(dir_path_c):
                os.makedirs(dir_path_c)
            new_file = dir_path_c + '\\' + new_dir + '_%d.jpg' % i
            '''
            c = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
            print(c.shape)
            h = c.shape[0]
            w = c.shape[1]
            for i in range(h):
                for j in range(w):
                    gray = c[i][j][0]
                    if gray < 85:
                        c[i][j][2] = 0
                        c[i][j][1] = 0
                        c[i][j][0] = gray * 3
                    elif gray < 170:
                        c[i][j][2] = 0
                        c[i][j][1] = (gray - 85) * 3
                        c[i][j][0] = 255 - (gray - 85) * 3
                    else:
                        c[i][j][2] = (gray - 170) * 3
                        c[i][j][1] = 255 - (gray - 170) * 3
                        c[i][j][0] = 0
            cv2.imwrite(new_file, c)
            color_list.append(c)
            '''
            color_image = cv2.applyColorMap(a, cv2.COLORMAP_JET)
            cv2.imwrite(new_file, color_image)
            color_list.append(color_image)
        file_list_2.append(color_list)
    print(len(file_list_2))
    return file_list_2

def hsv_images(in_path):
    file_list_3 = []
    for filename in os.listdir(in_path):
        hsv_list = []
        spectral = hdf5.loadmat(in_path + "\\" + filename)
        hsv_path = in_path + '_hsv'
        if not os.path.exists(hsv_path):
            os.makedirs(hsv_path)
        for i in range(31):
            img = spectral['cube'][:, :, i]
            a = (img * 255).astype(np.uint8)
            new_dir = filename.replace('.mat', '')
            dir_path_h = hsv_path + '\\' + new_dir
            if not os.path.exists(dir_path_h):
                os.makedirs(dir_path_h)
            new_file = dir_path_h + '\\' + new_dir + '_%d.jpg'%i
            c = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
            HSV = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
            h = HSV.shape[0]
            w = HSV.shape[1]
            for k in range(h):
                for s in range(w):
                    h = (180 + 155 - i * 6) % 180
                    HSV[k][s][0] = h
                    HSV[k][s][1] = 255 - HSV[k][s][2]
            #print(HSV)
            HSV = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
            cv2.imwrite(new_file, HSV)
            hsv_list.append(a)
        file_list_3.append(hsv_list)
    print(len(file_list_3))
    return file_list_3

if __name__ == "__main__":

    in_path = './ensemble/track1'
    #file_list_1 = split_images(in_path)
    #file_list_2 = color_images(in_path)
    color_images(in_path)
