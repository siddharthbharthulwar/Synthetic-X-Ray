import PIL
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

print(cv)

def discrete_radon_transform(image, steps):
    R = np.zeros((steps, len(image)), dtype='float64')
    for s in range(steps):
        rotation = misc.imrotate(image, -s*180/steps).astype('float64')
        R[:,s] = sum(rotation)
    return R

# Read image as 64bit float gray scale
#image = misc.imread('slice.png', flatten=True).astype('float64')
image = cv.imread('XrayGeneration\slice.png', 0).astype('float64')
radon = discrete_radon_transform(image, 220)

# Plot the original and the radon transformed image
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(radon, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()