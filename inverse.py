import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = r"Data\In\3\0877.png"

img = cv.imread(path, cv.IMREAD_GRAYSCALE)
plt.imsave('inverted_2.png', 1 - img, cmap = 'gray')