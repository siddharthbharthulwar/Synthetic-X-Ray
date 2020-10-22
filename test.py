import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Data/In/0/0001.png', cv.IMREAD_GRAYSCALE)
img = img / 255
img = 1 - img
plt.imshow(img, cmap = 'gray')
plt.show()

plt.imsave('inverted.png', img, cmap = 'gray')