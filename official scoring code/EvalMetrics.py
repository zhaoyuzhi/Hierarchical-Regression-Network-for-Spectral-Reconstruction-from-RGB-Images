import numpy as np
from sklearn.cluster import MiniBatchKMeans

def computeMRAE(groundTruth, recovered):
    """
    Compute MRAE between two images
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :return: Mean Realative Absolute Error between `recovered` and `groundTruth`.
    """
    
    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"
    
    difference = np.abs(groundTruth - recovered) / groundTruth
    mrae = np.mean(difference) 

    return mrae


def computeRMSE(groundTruth, recovered):
    """
    Compute RMSE between two images
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :return: RMSE between `recovered` and `groundTruth`.
    """

    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"

    difference = (groundTruth - recovered) ** 2
    rmse = np.sqrt(np.mean(difference))

    return rmse


def evalBackProjection(groundTruth, recovered, cameraResponse):
    """
    Score the colorimetric accuracy of a recovered spectral image vs. a ground truth reference image.
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :param cameraResponse: camera response functions. (Spectral_Dimension x RGB_Dimension)
    :return: MRAE between ground-truth and recovered RGBs.
    """

    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"
    assert groundTruth.shape[2] == cameraResponse.shape[0], "Spectral dimension mismatch between spectral images and camera response functions"
    
    specDim = cameraResponse.shape[0] # spectral dimension
    
    # back projection + reshape the data into num_of_samples x spectral_dimensions
    groundTruthRGB = np.matmul(groundTruth.reshape(-1, specDim), cameraResponse)
    recoveredRGB = np.matmul(recovered.reshape(-1, specDim), cameraResponse)
    
    # calculate MRAE
    difference = np.abs(groundTruthRGB - recoveredRGB) / groundTruthRGB
    mrae = np.mean(difference)

    return mrae


def labelPixelGroup(groundTruth, numberOfGroups=1000):
    """
    Use k-means to group similar spectra, and label the pixels with the group numbers.
    Note that k-means are calculated on normalized spectra (regardless of the intensity level)
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param numberOfGroups: number of representative spectra.
    :return: pixel labels that record to which group the spectrum at each pixel belongs. (Height x Width)
    """
    
    height, width, specDim = groundTruth.shape
    
    # reshape the data into num_of_samples x spectral dimensions, and normalize the spectra
    groundTruthList = groundTruth.reshape(-1, specDim)
    normalizedGroundTruthList = groundTruthList / np.linalg.norm(groundTruthList, axis=1, keepdims=True)

    # kmeans calculation best in n_init trials (mini batch kmeans approximation with batch size at 10% image size)
    batchSize = int(height * width * 0.1)
    trials = 5
    kmeans = MiniBatchKMeans(n_clusters=numberOfGroups, batch_size=batchSize, n_init=trials).fit(normalizedGroundTruthList)
    
    labeledImage = kmeans.labels_
    labeledImage = labeledImage.reshape(height, width)
    
    return labeledImage
    
    
def weightedAccuracy(groundTruth, recovered, labeledImage):
    """
    Compute the mean group performance in MRAE. Spectra are grouped by the ``labelPixelGroup'' function
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :param labeledImage: labeled image, output of ``labelPixelGroup'' function. (Height x Width)
    :return: mean group performance in MRAE
    """
    
    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"
    assert groundTruth.shape[:2] == labeledImage.shape[:2], "Size not match for spectral and labeled images"
    
    specDim = groundTruth.shape[2] # spectral dimension
    
    # reshape the inputs into num_of_samples x spectral_dimensions
    groundTruthList = groundTruth.reshape(-1, specDim)
    recoveredList = recovered.reshape(-1, specDim)
    labelList = labeledImage.reshape(-1)
    
    # list of group numbers
    groups = np.sort(np.unique(labelList)).astype(int)
    
    allMrae = []  # used to collect mrae of all groups
    
    # group by group calculating mean MRAE
    for groupNum in groups:
        groupPixels = labelList == groupNum
        
        groupGroundTruth  = groundTruthList[groupPixels, :]
        groupRecovered = recoveredList[groupPixels, :]
        
        # calculate MRAE
        groupDiff = np.abs(groupGroundTruth - groupRecovered) / groupGroundTruth
        groupMrae = np.mean(groupDiff)
        
        allMrae.append(groupMrae)
        
    print('Worst group: ', np.max(allMrae))    # worst performing group
    print('Best group:  ', np.min(allMrae))    # best performing group
    print('Mean:        ', np.mean(allMrae))   # mean group performance
    
    return np.mean(allMrae)   # mean group performance
    
    
def weightedBackProjectionAccuracy(groundTruth, recovered, cameraResponse, labeledImage):   
    """
    Compute the mean group performance of back projection accuracy. Spectra are grouped by the ``labelPixelGroup'' function
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :param cameraResponse: camera response functions. (Spectral_Dimension x RGB_Dimension)
    :param labeledImage: labeled image, output of ``labelPixelGroup'' function. (Height x Width)
    :return: mean group performance in MRAE
    """
    
    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"
    assert groundTruth.shape[:2] == labeledImage.shape[:2], "Size not match for spectral and labeled images"
    assert groundTruth.shape[2] == cameraResponse.shape[0], "Spectral dimension mismatch between spectral images and camera response functions"
    
    specDim = cameraResponse.shape[0] # spectral dimension
    
    # back projection + reshape the data into num_of_samples x spectral_dimensions
    groundTruthRGB  = np.matmul(groundTruth.reshape(-1, specDim), cameraResponse)
    recoveredRGB = np.matmul(recovered.reshape(-1, specDim), cameraResponse)
    labelList = labeledImage.reshape(-1)
    
    # list of group numbers
    groups = np.sort(np.unique(labelList)).astype(int)
    
    allMrae = []  # used to collect mrae of all groups
    
    # group by group calculating mean MRAE
    for groupNum in groups:
        groupPixels = labelList == groupNum
        
        groupGroundTruth  = groundTruthRGB[groupPixels, :]
        groupRecovered = recoveredRGB[groupPixels, :]
        
        # calculate MRAE
        groupDiff = np.abs(groupGroundTruth - groupRecovered) / groupGroundTruth
        groupMrae = np.mean(groupDiff)
        
        allMrae.append(groupMrae)
        
    print('Worst group: ', np.max(allMrae))    # worst performing group
    print('Best group:  ', np.min(allMrae))    # best performing group
    print('Mean:        ', np.mean(allMrae))   # mean group performance
    
    return np.mean(allMrae)   # mean group performance
