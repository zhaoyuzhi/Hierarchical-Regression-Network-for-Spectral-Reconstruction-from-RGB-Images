import argparse
import os
import cv2
import numpy as np
import hdf5storage as hdf5
from scipy.io import loadmat
from matplotlib import pyplot as plt

from SpectralUtils import savePNG, projectToRGB
from EvalMetrics import computeMRAE

BIT_8 = 256

# read path
def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if filespath[-4:] == '.mat':
                ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if filespath[-4:] == '.mat':
                ret.append(filespath)
    return ret

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def demo_track1(filePath, filtersPath):
    
    #filePath = "F:\\NTIRE 2020\\spectral reconstruction\\code1\\en4_track1\\ARAD_HS_0451.mat"
    #filtersPath = "./resources/cie_1964_w_gain.npz"

    # Load HS image and filters
    cube = hdf5.loadmat(filePath)['cube']
    #cube = loadmat(filePath)['cube']
    filters = np.load(filtersPath)['filters']

    # Project image to RGB
    rgbIm = np.true_divide(projectToRGB(cube, filters), BIT_8)

    # Save image file
    path = 'temp_clean.png'
    savePNG(rgbIm, path)

    # Display RGB image
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.title('Example "Clean" Output Image')
    plt.show()

def single_img_mrae(generated_mat_path, groundtruth_mat_path):
    #generated_mat_path = "F:\\NTIRE 2020\\spectral reconstruction\\code1\\en4_track1\\ARAD_HS_0451.mat"
    #groundtruth_mat_path = "F:\\NTIRE 2020\\spectral reconstruction\\NTIRE2020_Validation_Spectral\\ARAD_HS_0451.mat"
    generated_mat = hdf5.loadmat(generated_mat_path)['cube'] # shape: (482, 512, 31)
    groundtruth_mat = hdf5.loadmat(groundtruth_mat_path)['cube'] # shape: (482, 512, 31)
    mrae = computeMRAE(generated_mat, groundtruth_mat)
    print(mrae)
    return mrae

def folder_img_mrae(generated_folder_path, groundtruth_folder_path):
    #generated_folder_path = "F:\\NTIRE 2020\\spectral reconstruction\\code1\\en4_track1"
    #groundtruth_folder_path = "F:\\NTIRE 2020\\spectral reconstruction\\NTIRE2020_Validation_Spectral"
    matlist = get_jpgs(generated_folder_path)
    avg_mrae = 0
    for i, matname in enumerate(matlist):
        generated_mat_path = os.path.join(generated_folder_path, matname)
        groundtruth_mat_path = os.path.join(groundtruth_folder_path, matname)
        generated_mat = hdf5.loadmat(generated_mat_path)['cube'] # shape: (482, 512, 31)
        groundtruth_mat = hdf5.loadmat(groundtruth_mat_path)['cube'] # shape: (482, 512, 31)
        mrae = computeMRAE(generated_mat, groundtruth_mat)
        avg_mrae = avg_mrae + mrae
        print('The %d-th mat\'s mrae:' % (i + 1), mrae)
    avg_mrae = avg_mrae / len(matlist)
    print('The average mrae is:', avg_mrae)
    return avg_mrae

generated_folder_path = "F:\\NTIRE 2020\\spectral reconstruction\\ensemble\\ensemble\\track1"
generated_folder_path = "F:\\NTIRE 2020\\spectral reconstruction\\ensemble\\ensemble\\track2"

groundtruth_folder_path = "F:\\NTIRE 2020\\spectral reconstruction\\NTIRE2020_Validation_Spectral"
avg_mrae = folder_img_mrae(generated_folder_path, groundtruth_folder_path)
