import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from copy import deepcopy
import numpy.ma as ma
from math import pi, sqrt
from skimage.draw import circle

def get_first(slice, target):

    for i in range(0, len(slice)):

        if (slice[i] == target):

            return i

def get_last(slice, target):

    for i in range(len(slice) - 1, 0, -1):

        if (slice[i] == target):
            
            return i

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

def extract_lung_mask(volume, size = 128):

    slices = []

    for i in range(0, size):

        masked_slice = extract_slice_mask(volume[i])
        slices.append(masked_slice)

    slices = np.array(slices)
    return slices

def tpfp(truth, prediction):

    h, w, l = truth.shape[0], truth.shape[1], truth.shape[2]

    tp = np.zeros(truth.shape)
    fp = np.zeros(truth.shape)
    fn = np.zeros(truth.shape)
    tn = np.zeros(truth.shape)

    truth_mask = extract_lung_mask(truth)
    pred_mask = extract_lung_mask(prediction)

    for i in range(0, h):

        for j in range(0, w):

            for k in range(0, l):

                if (truth_mask[i, j, k] == 1 and pred_mask[i, j, k] == 1):

                    tp[i, j, k] = 1

                elif (truth_mask[i, j, k] == 1 and pred_mask[i, j, k] == 0):

                    fn[i, j, k] = 1

                elif (truth_mask[i, j, k] == 0 and pred_mask[i, j, k] == 1):

                    fp[i, j, k] = 1

                else:

                    tn[i, j, k] = 1

    tp = ma.masked_values(tp * 100, 0)
    fp = ma.masked_values(fp * 50, 0)
    fn = ma.masked_values(fn, 0)
    
    prdd = pred_mask[64]
    trdd = truth_mask[64]

    return tp, fp, fn

class WEDClass:

    def __init__(self, truth, view1, view2):
        
        self.truth = truth
        self.single = view1
        self.double = view2

        self.truth_cntslices = []
        self.truth_areas = []
        self.truth_radii = []
        self.truth_centroids = []

        
        for i in range(0, 128):

                tslice = self.truth[i]

                thresh = extract_slice_mask(tslice).filled(0)

                contours, hierarchy = cv.findContours(thresh.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            
                M = cv.moments(contours[0])
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                self.truth_centroids.append((cx, cy))
                area = cv.contourArea(contours[0])
                self.truth_areas.append(area)
                radius = sqrt(area / pi)
                self.truth_radii.append(radius)
            

                if len(contours) != 0:
                    temp = cv.fillPoly(tslice, pts = np.array(contours[0]), color = (255, 255, 255))
                    self.truth_cntslices.append(temp)

        self.single_cntslices = []
        self.single_areas = []
        self.single_radii = []
        self.single_centroids = []
        
        for i in range(0, 128):

                tslice = self.truth[i]

                thresh = extract_slice_mask(tslice).filled(0)

                contours, hierarchy = cv.findContours(thresh.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

                M = cv.moments(contours[0])
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                self.single_centroids.append((cx, cy))
                area = cv.contourArea(contours[0])
                self.single_areas.append(area)
                radius = sqrt(area / pi)
                self.single_radii.append(radius)

                if len(contours) != 0:
                    temp = cv.fillPoly(tslice, pts = np.array(contours[0]), color = (255, 255, 255))
                    self.single_cntslices.append(temp)

        self.double_cntslices = []
        self.double_areas = []
        self.double_radii = []
        self.double_centroids = []
        
        for i in range(0, 128):

                tslice = self.truth[i]

                thresh = extract_slice_mask(tslice).filled(0)

                contours, hierarchy = cv.findContours(thresh.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

                M = cv.moments(contours[0])
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                self.double_centroids.append((cx, cy))
                area = cv.contourArea(contours[0])
                self.double_areas.append(area)
                radius = sqrt(area / pi)
                self.double_radii.append(radius)

                if len(contours) != 0:
                    temp = cv.fillPoly(tslice, pts = np.array(contours[0]), color = (255, 255, 255))
                    self.single_cntslices.append(temp)
        
    def circles(self):

        #note: this is only with the truth version right now

        self.truth_circle = []

        for i in range(0, 128):

            radius = self.truth_radii[i]
            coords = self.truth_centroids[i]
            rr, cc = circle(coords[0], coords[1], radius)
            tempslice = np.zeros((128, 128))
            tempslice[rr, cc] = 1
            self.truth_circle.append(tempslice)

        self.truth_circle = np.array(self.truth_circle)

        self.single_circle = []

        for i in range(0, 128):

            radius = self.truth_radii[i]
            coords = self.truth_centroids[i]
            rr, cc = circle(coords[0], coords[1], radius)
            tempslice = np.zeros((128, 128))
            tempslice[rr, cc] = 1
            self.single_circle.append(tempslice)

        self.single_circle = np.array(self.single_circle)

        self.double_circle = []

        for i in range(0, 128):

            radius = self.truth_radii[i]
            coords = self.truth_centroids[i]
            rr, cc = circle(coords[0], coords[1], radius)
            tempslice = np.zeros((128, 128))
            tempslice[rr, cc] = 1
            self.double_circle.append(tempslice)

        self.double_circle = np.array(self.double_circle)

    def get_boundaries(self, index):

        self.truth_boundaries = []
        
        for i in range(0, 128):

            slicc = self.truth_circle[i][index]
            tupp = (get_first(slicc, 1), get_last(slicc, 1))
            self.truth_boundaries.append(tupp)

        self.single_boundaries = []
        
        for i in range(0, 128):

            slicc = self.single_circle[i][index]
            tupp = (get_first(slicc, 1), get_last(slicc, 1))
            self.single_boundaries.append(tupp)

        self.double_boundaries = []
        
        for i in range(0, 128):

            slicc = self.double_circle[i][index]
            tupp = (get_first(slicc, 1), get_last(slicc, 1))
            self.double_boundaries.append(tupp)

    def final_point_creation(self):

        self.truth_left_finals = []
        self.single_left_finals = []
        self.double_left_finals = []

        self.truth_right_finals = []
        self.single_right_finals = []
        self.double_right_finals = []

        for i in range(0, 128):

            self.truth_left_finals.append(self.truth_boundaries[i][0])
            self.truth_right_finals.append(self.truth_boundaries[i][1])
            self.single_left_finals.append(self.truth_boundaries[i][0])
            self.single_right_finals.append(self.single_boundaries[i][1])
            self.double_left_finals.append(self.truth_boundaries[i][0])
            self.double_right_finals.append(self.double_boundaries[i][1])


    def plot_all(self):

        f = plt.figure(figsize=(16, 6))

        f.add_subplot(1, 3, 1)
        plt.imshow(self.truth[:, 64], cmap = 'gray')
        plt.plot(range(0, 128), self.truth_left_finals, 'g')
        plt.plot(range(0, 128), self.truth_right_finals, 'g')

        f.add_subplot(1, 3, 2)
        plt.imshow(self.truth[:, 64], cmap = 'gray')
        plt.plot(range(0, 128), self.single_left_finals, 'b')
        plt.plot(range(0, 128), self.single_right_finals, 'b')

        f.add_subplot(1, 3, 3)
        plt.imshow(self.truth[:, 64], cmap = 'gray')
        plt.plot(range(0, 128), self.double_left_finals, 'r')
        plt.plot(range(0, 128), self.double_right_finals, 'r')

        plt.show()




    


        

    