import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy.ma as ma
from dataset import PairedDatasetSingle

'''
array = np.load('Data/Out_New/0001.npy')

plt.imshow(array[64])
plt.show()
'''

def extract_slice_mask(slice):

    ret, thresh = cv.threshold(slice, 0.1, 1, cv.THRESH_BINARY)
    num_labels, labels_im, stats, centroids = cv.connectedComponentsWithStats(thresh.astype('uint8'))

    max = stats[1, 4]
    largest_index = 1

    for k in np.delete(np.unique(labels_im), 0):

        size = stats[k, 4]
        if (size > max):

            max = size
            largest_index = k

    mask = ma.masked_not_equal(labels_im, largest_index)
    return mask

pd = PairedDatasetSingle('Data/In/0', 'Data/Out_New', 10)

contour_slices = []


for i in range(0, 127):

    tslice = pd.y_train[0][i]

    thresh = extract_slice_mask(tslice).filled(0)

    im2, contours, hierarchy = cv.findContours(thresh.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


    if len(contours) != 0:

        print(len(contours))
        #cv.drawContours(tslice, contours, -1, 255, 3)
        yeet = cv.fillPoly(tslice, pts = np.array(contours[0]), color = (255, 255, 255))
        contour_slices.append(yeet)


contour_slices = np.array(contour_slices)
print(contour_slices.shape)

for i in range(0, 127):

    plt.imshow(contour_slices[i])
    plt.show()