import os
import random
import numpy as np
import cv2
import scipy.io as io
import torch
from torch.utils.data import Dataset

import utils

class HS_multiscale_DSet(Dataset):
    def __init__(self, opt):                                   		    # root: list ; transform: torch transform
        self.opt = opt
        # the root of both domains
        self.baseroot_A = os.path.join(opt.baseroot, 'NTIRE2020_Train_Spectral')
        if opt.save_path == 'track1':
            self.baseroot_B = os.path.join(opt.baseroot, 'NTIRE2020_Train_Clean')
        if opt.save_path == 'track2':
            self.baseroot_B = os.path.join(opt.baseroot, 'NTIRE2020_Train_RealWorld')
        namelist = self.get_names(self.baseroot_A)
        # build image list
        self.imglist_A, self.imglist_B = self.build_imglist(namelist, opt)
    
    def get_names(self, path):
        # read a folder, return the image name
        ret = []
        for root, dirs, files in os.walk(path):  
            for filespath in files: 
                ret.append(filespath[:12])          # e.g. ARAD_HS_0001
        return ret
        
    def build_imglist(self, namelist, opt):
        # build an imglist
        imglist_A = []
        imglist_B = []
        for i in range(len(namelist)):
            imglist_A.append(namelist[i] + '.mat')
            if opt.save_path == 'track1':
                imglist_B.append(namelist[i] + '_clean.png')
            if opt.save_path == 'track2':
                imglist_B.append(namelist[i] + '_RealWorld.jpg')
        return imglist_A, imglist_B

    def __getitem__(self, index):
        # read an image
        imgpath_A = os.path.join(self.baseroot_A, self.imglist_A[index])
        img_A = io.loadmat(imgpath_A)['cube']       # (482, 512, 31), in range [0, 1], float64
        imgpath_B = os.path.join(self.baseroot_B, self.imglist_B[index])
        img_B = cv2.imread(imgpath_B, -1)           # (482, 512, 3), in range [0, 255], uint8

        # normalization
        img_B = img_B.astype(np.float64) / 255.0

        # crop
        if self.opt.crop_size > 0:
            h, w = img_A.shape[:2]
            rand_h = random.randint(0, h - self.opt.crop_size)
            rand_w = random.randint(0, w - self.opt.crop_size)
            img_A = img_A[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]    # (256, 256, 31), in range [0, 1], float64
            img_B = img_B[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]    # (256, 256, 3), in range [0, 1], float64

        # to tensor
        img_A = torch.from_numpy(img_A.astype(np.float32).transpose(2, 0, 1)).contiguous()
        img_B = torch.from_numpy(img_B.astype(np.float32).transpose(2, 0, 1)).contiguous()

        return img_B, img_A

    def __len__(self):
        return len(self.imglist_A)

class HS_multiscale_ValDSet(Dataset):
    def __init__(self, opt):                                   		    # root: list ; transform: torch transform
        self.opt = opt
        # the root of both domains
        self.baseroot = os.path.join(opt.baseroot)
        # build image list
        self.imglist = utils.get_jpgs(self.baseroot)

    def __getitem__(self, index):
        # read an image
        imgname = self.imglist[index]
        imgpath = os.path.join(self.baseroot, imgname)
        img = cv2.imread(imgpath, -1)           # (482, 512, 3), in range [0, 255], uint8

        # normalization
        img = img.astype(np.float64) / 255.0

        # crop
        '''
        h, w = img.shape[:2]
        rand_h = random.randint(0, h - self.opt.crop_size)
        rand_w = random.randint(0, w - self.opt.crop_size)
        '''
        img1 = img[0:480, 0:512, :]
        img2 = img[2:482, 0:512, :]

        # to tensor
        img1 = torch.from_numpy(img1.astype(np.float32).transpose(2, 0, 1)).contiguous()
        img2 = torch.from_numpy(img2.astype(np.float32).transpose(2, 0, 1)).contiguous()

        return img1, img2, imgname

    def __len__(self):
        return len(self.imglist)
