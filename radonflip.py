import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import shutil
from scipy.ndimage import zoom
import scipy.ndimage 
from skimage.transform import radon

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

def isFlipped(path):

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    fig = plt.figure()
    plt.imshow(img, cmap = 'gray')
    plt.title(path)

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    val = input("F: Flip")
    print(val)

    if (val == 'f'):

        return True

    else:

        return False

def testslice(array):

    plt.imshow(array[:, 100, :], cmap = 'gray')
    plt.show()

def flip_ct(array):

    flipped = np.rot90(array, k = 2, axes = (2, 0))
    return flipped

def flip_xr(array):

    flipped = np.rot90(array, k = 2)
    return flipped

def move(path, outpath):

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    cv.imwrite(outpath, img)

root = 'Data\Out'

for i in os.listdir(root):

    xray = createXRay(os.path.join(root, i), 0)
    plt.imsave(os.path.join('Data\FlipControl', i[0:4] + '.png'), xray, cmap = 'gray')
    print(i)

#write crawler to manually inspect all of them 