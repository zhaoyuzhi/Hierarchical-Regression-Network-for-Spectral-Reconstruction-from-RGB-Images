import hdf5storage as hdf5
import numpy as np
import scipy.io as io
import h5py

data1 = np.random.randn(4, 4)

hdf5.write(data = data1, path = 'cube', filename = './data1.mat', matlab_compatible=True)
'''
data = h5py.File('./data1.mat')
print(data.keys())
'''
print(data1)

data = hdf5.loadmat('./data1.mat')
print(data)
