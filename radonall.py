import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import radon, iradon
from scipy.ndimage import zoom
import os
import pydicom as dicom
import scipy.ndimage
import numpy as np
import cv2 as cv

def radonTransformation(img, angle):

    img = zoom(img, 0.4)

    projections = radon(img, theta = [angle])
    return projections


def createXRay(volumePath, angle):

    b = np.load(volumePath)

    b = np.rot90(b, k = 2, axes = (2, 0))

    ls = []

    for sl in b:

        ls.append(radonTransformation(sl, angle))

    ls = np.array(ls)

    return np.rot90(np.squeeze(ls), 2)

path = r"Data\Out"
ANGLE = 0

for item in os.listdir(path):

    array = createXRay(os.path.join(path, item), ANGLE)
    filename = item[0:3]
    plt.imshow(array, cmap = 'gray')
    plt.title(filename)
    plt.show()