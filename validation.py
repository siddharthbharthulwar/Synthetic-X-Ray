import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
#from dataset import PairedDatasetDouble, PairedDatasetQuad, PairedDatasetSingle
import numpy.ma as ma
import scipy.ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, morphology

class ValidationDataset:

    def __init__(self):

        self.truths = []
        self.predictions = []

        

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
'''
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