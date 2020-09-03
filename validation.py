import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
#from dataset import PairedDatasetDouble, PairedDatasetQuad, PairedDatasetSingle
import numpy.ma as ma
import scipy.ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, morphology
import os
from tensorflow.keras.models import model_from_json
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error

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


class ValidationDataset:

    def __init__(self, truth_out_dir, pred_out_dir, model_json_path, model_weights_path):
        
        with open(model_json_path, 'r') as f:

            self.model = model_from_json(f.read())

        self.model.load_weights(model_weights_path)

        self.truth_dirs = os.listdir(truth_out_dir)
        self.pred_dirs = os.listdir(pred_out_dir)

        self.truths = []
        self.preds = []

        for file in self.truth_dirs:

            if (file[-3:] == 'npy'):

                array = np.load(os.path.join(truth_out_dir, file))
                print("truth {}".format(file))
                self.truths.append(array)

        for file in self.pred_dirs:

            if (file[-3:] == 'npy'):

                array = np.load(os.path.join(pred_out_dir, file))
                print("pred {}".format(file))
                self.preds.append(array)

    def view_pair(self, index):

        truth = self.truths[index]
        prediction = self.preds[index]

        f = plt.figure()

        plt.imshow(self.truths[index][64], cmap = 'gray')
        plt.title("Truth {}".format(index))

        plt.imshow(self.preds[index][64], cmap = 'gray')
        plt.title("Pred {}".format(index))

        plt.imshow(block = True)

    def tpfp(self, index):

        truth = self.truths[index]
        prediction = self.preds[index]

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

        dice = np.sum(prediction[truths==1])*2.0 / (np.sum(prediction) + np.sum(truths))


        for i in range(0, h):

            plt.imshow(tp[i], cmap = 'brg', vmin = 0.1) #green
            plt.imshow(fp[i], cmap = 'brg', vmin = 0.1, vmax = 100) #red
            plt.imshow(fn[i], cmap = 'brg', vmin = 0.1, vmax = 90) #blue
            plt.title("{} w/ Dice {}".format(i, dice))
            plt.show(block = True)

    def compare_histograms(self, index):

        truth = self.truths[index]
        prediction = self.preds[index]

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(truth[64], cmap = 'gray')
        axs[0, 0].set_title("Truth {}".format(index))
        axs[0, 1].imshow(prediction[64], cmap = 'gray')
        axs[0, 1].set_title("Prediction {}".format(index))
        axs[1, 0].hist(truth.ravel(), bins = 400)
        axs[1, 0].set_title("Hist of Truth")
        axs[1, 1].hist(prediction.ravel(), bins = 400)
        axs[1, 1].set_title("Hist of Pred")

        plt.show(block = True)

    def compute_stats(self, nums):

        self.mses = []
        self.ssims = []

        if (nums > len(self.preds)):

            num = len(self.preds)

        for i in range(0, nums):

            truth = self.truths[i]
            prediction = self.preds[i]

            mse = mean_squared_error(truth, prediction)
            ssim = structural_similarity(truth, prediction, data_range = truth.max() - truth.min())

            self.mses.append(mse)
            self.ssims.append(ssim)

            print("Item {} with SSIM {} and MSE {}".format(i, ssim, mse))
    
