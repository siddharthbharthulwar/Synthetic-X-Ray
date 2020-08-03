import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import radon, iradon
from scipy.ndimage import zoom
import os
import pydicom as dicom
import scipy.ndimage
import numpy as np

def radonTransformation(img, angle):

    img = zoom(img, 0.4)

    projections = radon(img, theta = [angle])
    return projections


def createXRay(volumePath, angle):

    b = get_pixels_hu(load_scan(volumePath))

    ls = []

    for sl in b:

        ls.append(radonTransformation(sl, angle))

    ls = np.array(ls)

    return np.rot90(np.squeeze(ls), 2)

