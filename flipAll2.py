import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def testslice(array):

    plt.imshow(array[:, 100, :], cmap = 'gray')
    plt.show()

def flip_ct(array):

    flipped = np.rot90(array, k = 2, axes = (2, 0))
    return flipped

def flip_xr(array):

    flipped = np.rot90(array, k = 2)
    return flipped
