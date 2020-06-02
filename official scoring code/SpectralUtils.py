import os
import cv2 as cv
import numpy as np
from scipy.io import savemat


JPG_QUALITY = 80
PNG_COMPRESSION = 9  # Prioritize small file size over speed (lossless compression)


# Bit-depth constants
BIT_16 = 65535
BIT_8 = 255


def imTo8bit(im):
    """
    Reduce image to 8bit color
    :param im: RGB image in floating point format, values above 1.0 will be mapped to 255.
    :return: image in 8 bit format
    """
    assert im.min() >= 0, "Images must contain only non-negative values"
    return (np.clip(im, 0, 1) * BIT_8)


def imTo16bit(im):
    """
    Reduce image to 16bit color
    :param im: RGB image in floating point format, values above 1.0 will be mapped to 65535.
    :return: image in 16 bit format
    """
    assert im.min() >= 0, "Images must contain only non-negative values"
    return np.clip(im, 0, 1) * BIT_16

def projectToRGB(hsIm, cameraResponse):
    """
    Project a hyperspectral image to RGB using a known camera response
    :param hsIm: Hyperspectral image (H x W x C)
    :param cameraResponse: RGB camera response function (C x 3), channel order should be red, green, blue.
    :return: rgbIm: Resulting RGB image
    """
    rgbIm = np.dot(hsIm, cameraResponse)

    return rgbIm


def projectToRGBMosaic(hsIm, cameraResponse):
    """
    Create RGGB mosaic image from a hyperspectral image to RGB using a known camera response
    :param hsIm: Hyperspectral image (H x W x C)
    :param cameraResponse: RGB camera response function (C x 3), channel order should be red, green, blue.
    :return: mosaicIm: RGGB mosaic image
    """
    rgbIm = projectToRGB(hsIm, cameraResponse)
    mosaicIm = np.zeros(rgbIm.shape[:2])

    # Collect pixel
    mosaicIm[0::2, 0::2] = rgbIm[0::2, 0::2, 0]
    mosaicIm[1::2, 0::2] = rgbIm[1::2, 0::2, 1]
    mosaicIm[0::2, 1::2] = rgbIm[0::2, 1::2, 1]
    mosaicIm[1::2, 1::2] = rgbIm[1::2, 1::2, 2]

    return mosaicIm


def addNoise(mosaicIm, darkNoise = 10, targetNpe = 5000):
    """
    Add noise to a RGGB mosaic image produced by `projectToRGBMosaic` and return a 16bit noisy camera mosaic.
    :param mosaicIm: Input RGGB mosaic (H x W)
    :param targetNpe: Light intensity parameter: the Target average Npe in the green band  
    :param darkNoise: dark noise
    :return: Noisy RGGB mosaic (H x W)
    - Uses global variables:
    targetNpe:      Light intensity parameter: the Target Npe for the average of the green band, in [PhotoElectron] units
    darkNoise:      Dark noise level in [PhotoElectron] units
    BIT_16:   The max allowed digital number
    """
    # Find the mean of green channel used to approximate scene luminance level
    meanOverGreenPixels = np.mean([mosaicIm[1::2, 0::2], mosaicIm[0::2, 1::2]])
    
    # Set the scaling functions to Number of Photo-Electrons and back to digital number 
    AvgGreen2Npe = (targetNpe/ meanOverGreenPixels ) # input green Npe divided into 1nm bands
    analogGain= 1/AvgGreen2Npe


    # Scale the input image to target illumination levels. Result is in Npe units
    rggb_Npe = mosaicIm * AvgGreen2Npe # Scaling average greens to the target Npe for Poisson

    N_shot = np.random.poisson(rggb_Npe) # Randomize signal by poisson Shot noise model
    N_dark = np.random.normal(0, np.power(darkNoise, 2), rggb_Npe.shape) # Add normal Gaussian dark noise
    noisyRGGB = (N_shot + N_dark) # Total noisy signal

    # Ppply gain to scale the image into the correct DN values and apply A2D function (round and Clip to A2D range)
    digitizedRGGB = np.round(noisyRGGB*analogGain).clip(0,BIT_16) # Digitize signal, i.e., round and clip
    return digitizedRGGB


def demosaic(mosaicIm):
    """
    Demosaic RGGB "RAW" image, simulating a 16bit camera pipeline
    :param mosaicIm: RGGB mosaic image (H x W) in floating point or uint16 format.
    :return: 8bit RGB image (H x W x 3)
    """
    bgr = cv.cvtColor(mosaicIm.astype(np.uint16), cv.COLOR_BayerBG2BGR_EA)
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    return rgb / BIT_16


def saveJPG(im, path):
    """
    Save RGB image as 8bit compressed JPG. JPG compression setting defined by `JPG_QUALITY` constant
    :param im: RGB image in floating point format (H x W x 3).
    :param path: full path to image file name
    :return: True if successful
    """
    assert os.path.splitext(path)[1] == '.jpg', "File extension must be '.jpg'"

    # reduce image to 8 bit color
    im8 = imTo8bit(im).astype(np.uint8)

    imBGR = cv.cvtColor(im8, cv.COLOR_RGB2BGR)

    return cv.imwrite(path, imBGR, [cv.IMWRITE_JPEG_QUALITY, JPG_QUALITY])


def savePNG(im, path):
    """
    Save RGB image as 8bit PNG.
    :param im: RGB image in floating point format (H x W x 3).
    :param path: full path to image file name
    :return: True if successful
    """
    assert os.path.splitext(path)[1] == '.png', "File extension must be '.png'"

    # reduce image to 8 bit color
    im8 = imTo8bit(im).astype(np.uint8)
    imBGR = cv.cvtColor(im8, cv.COLOR_RGB2BGR)
    return cv.imwrite(path, imBGR, [cv.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])


def saveReconstructedHSI(hsIm, path):
    """
    Save the reconstructed hyperspectral image into .mat file.
    :param hsIm: hyperspectral image (H x W x S).
    :param path: full output path to image file name
    """
    
    # variable name = "cube"
    savemat(path, {"cube": hsIm})