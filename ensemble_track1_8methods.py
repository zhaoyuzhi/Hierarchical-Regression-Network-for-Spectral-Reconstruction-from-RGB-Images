import argparse
import os
import torch
import numpy as np
import cv2
import hdf5storage as hdf5

import utils

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--path1', type = str, default = './test/track1/0', \
            help = 'load the pre-trained model with certain epoch, None for pre-training')
    parser.add_argument('--path2', type = str, default = './test/track1/1', \
            help = 'load the pre-trained model with certain epoch, None for pre-training')
    parser.add_argument('--path3', type = str, default = './test/track1/2', \
            help = 'load the pre-trained model with certain epoch, None for pre-training')
    parser.add_argument('--path4', type = str, default = './test/track1/3', \
            help = 'load the pre-trained model with certain epoch, None for pre-training')
    parser.add_argument('--path5', type = str, default = './test/track1/4', \
            help = 'load the pre-trained model with certain epoch, None for pre-training')
    parser.add_argument('--path6', type = str, default = './test/track1/5', \
            help = 'load the pre-trained model with certain epoch, None for pre-training')
    parser.add_argument('--path7', type = str, default = './test/track1/6', \
            help = 'load the pre-trained model with certain epoch, None for pre-training')
    parser.add_argument('--path8', type = str, default = './test/track1/7', \
            help = 'load the pre-trained model with certain epoch, None for pre-training')
    parser.add_argument('--val_path', type = str, default = './ensemble', help = 'saving path that is a folder')
    parser.add_argument('--task_name', type = str, default = 'track1', help = 'task name for loading networks, saving, and log')
    # NTIRE2020_Validation_Clean    NTIRE2020_Validation_RealWorld
    opt = parser.parse_args()

    sample_folder = os.path.join(opt.val_path, opt.task_name)
    utils.check_path(sample_folder)

    imglist = utils.get_jpgs(opt.path1)
    for imgname in imglist:
        # Read
        data_path1 = os.path.join(opt.path1, imgname)
        data1 = hdf5.loadmat(data_path1)['cube']
        data_path2 = os.path.join(opt.path2, imgname)
        data2 = hdf5.loadmat(data_path2)['cube']
        data_path3 = os.path.join(opt.path3, imgname)
        data3 = hdf5.loadmat(data_path3)['cube']
        data_path4 = os.path.join(opt.path4, imgname)
        data4 = hdf5.loadmat(data_path4)['cube']
        data_path5 = os.path.join(opt.path5, imgname)
        data5 = hdf5.loadmat(data_path5)['cube']
        data_path6 = os.path.join(opt.path6, imgname)
        data6 = hdf5.loadmat(data_path6)['cube']
        data_path7 = os.path.join(opt.path7, imgname)
        data7 = hdf5.loadmat(data_path7)['cube']
        data_path8 = os.path.join(opt.path8, imgname)
        data8 = hdf5.loadmat(data_path8)['cube']
        data_avg = (data1 + data2 + data3 + data4 + data5 + data6 + data7 + data8) / 8
        # Save
        save_img_path = os.path.join(sample_folder, imgname)
        print(save_img_path)
        hdf5.write(data = data_avg, path = 'cube', filename = save_img_path, matlab_compatible = True)
