import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def isFlipped(path): #if image needs to be flipped, this returns true

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    ret, thresh = cv.threshold(img, 115, 1, cv.THRESH_OTSU)
    thresh = thresh[100:900, 150:850]

    sector1 = thresh[0:400, :]
    sector2 = thresh[400:800, :]

    sector1_nonzero = np.count_nonzero(sector1)
    sector2_nonzero = np.count_nonzero(sector2)

    print(sector2_nonzero - sector1_nonzero)

    if (sector2_nonzero > sector1_nonzero):

        return False

    else:

        return True

def flip180(path):

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return cv.rotate(img, cv.ROTATE_180)

path = r"D:\Documents\GitHub\GitHub\Synthetic-X-Ray\CXR\0002\0.png"

if (isFlipped(path)):

    plt.imshow(flip180(path))
    plt.show()