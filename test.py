import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

array = np.load('Data/Out/0001.npy')
array = np.clip(array, -1000, 3000)
sagittal = array[128]
coronal = array[:, 256]
axial = array[:, :, 256]

plt.imsave('sagittal.png', sagittal, cmap = 'gray', vmin = -1000, vmax = 3000)
plt.imsave('coronal.png', coronal, cmap = 'gray', vmin = -1000, vmax = 3000)
plt.imsave('axial.png', axial, cmap = 'gray', vmin = -1000, vmax = 3000)
#plt.imsave('inverted.png', img, cmap = 'gray')