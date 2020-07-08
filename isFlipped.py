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

def isFlipped2(path):

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    fig = plt.figure()
    plt.imshow(img, cmap = 'gray')
    plt.title(path)

    plt.draw()
    plt.waitforbuttonpress(0) # this will wait for indefinite time
    plt.close(fig)

    val = input("F for Flip, S for no Flip")

    if (val == 'F'):

        return 1

    elif (val == 'D'):

        return 3

    else:

        return 2

def flip180(path, bool):

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img = cv.rotate(img, cv.ROTATE_180)
    if (bool):
        cv.imwrite(path, img)
    else:

        plt.imshow(img, cmap = 'gray')
        plt.title(path)
        plt.show()
'''
path = r"D:\Documents\GitHub\GitHub\Synthetic-X-Ray\CXR\0002\0.png"

if (isFlipped(path)):

    plt.imshow(flip180(path))
    plt.show()
'''