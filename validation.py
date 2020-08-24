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

                array = np.load(os.path.join(self.truth_dirs, file))
                self.truths.append(array)

        for file in self.pred_dirs:

            if (file[-3:] == 'npy'):

                array = np.load(os.path.join(self.pred_dirs, file))
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

        for i in range(0, h):

            plt.imshow(tp[i], cmap = 'brg', vmin = 0.1) #green
            plt.imshow(fp[i], cmap = 'brg', vmin = 0.1, vmax = 100) #red
            plt.imshow(fn[i], cmap = 'brg', vmin = 0.1, vmax = 90) #blue
            plt.title(str(i))
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
'''



class ValidationDataset:

    def __init__(self, val_truth_path, val_pred_path):

        self.truth_path = val_truth_path
        self.pred_path = val_pred_path

        self.truth_files = os.listdir(self.truth_path)
        self.pred_files = os.listdir(self.pred_path)

        print("Length of truth files: {}".format(len(self.truth_files)))
        print("Length of prediction files: {}".format(len(self.pred_files)))

        self.truths = []
        self.predictions = []

        for item in self.truth_files:

            full_path = os.path.join(self.truth_path, item)
            arr = np.load(full_path)
            self.truths.append(arr)

        for item in self.pred_files:

            full_path = os.path.join(self.pred_path, item)
            arr = np.load(full_path)
            self.predictions.append(arr)

        

        
    def view_pair(self, index):

        f = plt.figure()
        plt.imshow(self.truths[index][64], cmap = 'gray')
        plt.title("Original CT {}".format(index))

        plt.imshow(self.predictions[index][64], cmap = 'gray')
        plt.title("Prediction CT {}".format(index))

        plt.show(block = True)



class ValidationPair:

    def __init__(self, truth_array, prediction_array, index):
        
        if (self.truth.shape == self.pred.shape):

            self.truth = truth_array
            self.pred = prediction_array
            self.index = index

        else:

            print("Error: expected equal shapes but received {} and {}".format(truth_array.shape, prediction_array.shape))


    def lung_segmentation(self):

        truth_masks = []

        for i in range(0, self.truth.shape[0]): #assuming volumes to be cubic

            mask = extract_slice_mask(self.truth[i])
            truth_masks.append(mask)
        
        pred_masks = []

        for i in range(0, self.pred.shape[0]):

            mask = extract_slice_mask(self.pred[i])
            pred_masks.append(mask)

        for i in range(0, len(truth_masks)):

            truth_mask = truth_masks[i]
            pred_mask = pred_masks[i]

            f = plt.figure()

            f.add_subplot(1, 2, 1)
            plt.imshow(truth_mask)
            f.add_subplot(1, 2, 2)
            plt.imshow(pred_mask)

            plt.show(block = True)



def plot3d(array):

    p = array.transpose(2, 1, 0)
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, 1)

    print(verts.shape, faces.shape)
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection = '3d')

    mesh = Poly3DCollection(verts[faces], alpha = 0.6)
    face_color = [0.45, 0.45, 0.75]
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    for ii in range(0,360,1):
        ax.view_init(elev=10., azim=ii)
        plt.savefig("movie{}.png".format(ii))

pd = PairedDatasetSingle('Data/In/0', 'Data/Out_New', 100)

for item in pd.y_train:

    masked_slices = []

    for i in range(0, 128):

        slice = extract_slice_mask(item[i])
        masked_slices.append(slice)

    plt.imshow(masked_slices[64])
    plt.show()
    masked_slices = np.array(masked_slices)

    plot3d(masked_slices)
'''