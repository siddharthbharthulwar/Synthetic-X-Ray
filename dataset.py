import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import os
import cv2 as cv

def normalize(array):

    normalized = (array - np.amin(array)) / (np.amax(array) - np.amin(array))

    return normalized

class PairedDatasetSingle:

    def __init__(self, in_path, out_path, filelimiter): #if filelimiter is -1 or > 943, unlimited.
        
        self.in_path = in_path
        self.out_path = out_path

        self.x_train = []
        self.y_train = []

        self.x_train_files = os.listdir(in_path)
        self.y_train_files = os.listdir(out_path)

        for i in range(0, len(self.x_train_files)):

            self.x_train_files[i] = self.x_train_files[i][0:4]

        for i in range(0, len(self.y_train_files)):

            self.y_train_files[i] = self.y_train_files[i][0:4]

        print("Length of x_train files: {}".format(len(self.x_train_files)))
        print("Length of y_train files: {}".format(len(self.y_train_files)))

        main_list = np.setdiff1d(self.x_train_files, self.y_train_files)

        for item in main_list:

            if (item[0] != '.'):

                if item in x_train_files:

                    os.remove(os.path.join(in_path, item + ".png"))
                
                elif item in y_train_files:

                    os.remove(os.path.join(out_path, item + ".npy"))

        # ^^ removing files present in one (x, y) and not other (x, y) to prevent dataset misalignment

        self.x_train_paths = []
        self.y_train_paths = []

        for item in os.listdir(in_path):

            if (item[-3: ] == 'png'):

                self.x_train_paths.append(item)

        for item in os.listdir(out_path):

            if (item[-3: ] == 'npy'):

                self.y_train_paths.append(item)

        self.x_train_paths = sorted(self.x_train_paths)
        self.y_train_paths = sorted(self.y_train_paths)

        self.keys = []

        for file in self.y_train_paths:

            self.keys.append(file[0:4])

        print("Length of keys: {}".format(len(self.keys)))

        if (filelimiter < 1 or filelimiter> len(self.x_train_paths)):

            filelimiter = len(self.x_train_paths)

        for i in range(0, filelimiter):

            pt_in = os.path.join(in_path, self.x_train_paths[i])
            pt_out = os.path.join(out_path, self.y_train_paths[i])

            self.x_train.append(cv.imread(pt_in, cv.IMREAD_GRAYSCALE))
            self.y_train.append(np.load(pt_out))

        self.x_train = np.array(self.x_train).astype('float64')
        self.x_train /= np.amax(self.x_train)
        
        for i in range(0, len(self.y_train)):

            item = self.y_train[i]
            item = np.clip(item, -1000, 3000)
            print("Item {} with max: {} and min: {}".format(i, np.amax(self.y_train[i]), np.amin(self.y_train[i])))
            normalized = normalize(item)
            self.y_train[i] = normalized
            print("Normalized item {} with max: {} and min: {}".format(i, np.amax(self.y_train[i]), np.amin(self.y_train[i])))
        
        self.y_train = np.array(self.y_train).astype('float64')


    def view_pair(self, index):

        f = plt.figure()

        f.add_subplot(1, 2, 1)
        plt.imshow(self.x_train[index], cmap = 'gray')
        plt.title('Radiograph of {}'.format(index))
        f.add_subplot(1, 2, 2)
        plt.imshow(self.y_train[index][64], cmap = 'gray')
        plt.title("Tomography of {}".format(index))
        plt.show(block = True)

    def view_animation(self, index):

        if (self.y_train):

            array = self.y_train[index]

            ims = []

            for im in array:

                gg = plt.imshow(im, cmap = 'gray')
                ims.append([gg])

            fig = plt.figure()
            ani = animation.ArtistAnimation(fig, ims, interval = 50, blit = True)
            plt.title(index)
            plt.show()

        else:

            print("Error: dataset has not been initialized yet")

    def prepare(self, num_train):

        self.x_val = self.x_train[num_train: len(self.x_train)]
        self.x_train = self.x_train[0:num_train]


        self.y_val = self.y_train[num_train : len(self.y_train)]
        self.y_train = self.y_train[0: num_train]


        self.x_train = np.reshape(self.x_train, (len(self.x_train), 1024, 1024, 1))
        self.y_train = np.reshape(self.y_train, (len(self.y_train), 128, 128, 128, 1))

        self.x_val = np.reshape(self.x_val, (len(self.x_val), 1024, 1024, 1))
        self.y_val = np.reshape(self.y_val, (len(self.y_val), 128, 128, 128, 1))

        print("Shape of X Train: {}".format(self.x_train.shape))
        print("Shape of Y Train: {}".format(self.y_train.shape))
        print("Shape of X Validation: {}".format(self.x_val.shape))
        print("Shape of Y Validation: {}".format(self.y_val.shape))

    def save_data(self, model, truth_out_dir, pred_out_dir, save_truths = False):

        self.model = model #tensorflow.keras model

        for i in range(0, len(self.keys)):

            input = np.reshape(self.x_train[i], (1, 1024, 1024, 1))
            output = model.predict(input)
            output = np.reshape(output, (128, 128, 128))

            if (save_truths):

                truth = np.reshape(self.y_train[i], (128, 128, 128))
                np.save(os.path.join(truth_out_dir, self.keys[i]), truth)

            np.save(os.path.join(pred_out_dir, self.keys[i]), output)
            print(self.keys[i])
    
class PairedDatasetDouble: #dataset for double view neural network model

    def __init__(self, in_path_0, in_path_1, out_path, filelimiter): #if filelimiter is -1 or > 943, unlimited.
        
        self.in_path_0 = in_path_0
        self.in_path_1 = in_path_1
        self.out_path = out_path

        self.x_train_0 = []
        self.x_train_1 = []
        self.y_train = []

        self.x_train_0_files = os.listdir(in_path_0)
        self.x_train_1_files = os.listdir(in_path_1)
        self.y_train_files = os.listdir(out_path)

        for i in range(0, len(self.x_train_0_files)):

            self.x_train_0_files[i] = self.x_train_0_files[i][0:4]

        for i in range(0, len(self.x_train_1_files)):

            self.x_train_1_files[i] = self.x_train_1_files[i][0:4]

        for i in range(0, len(self.y_train_files)):

            self.y_train_files[i] = self.y_train_files[i][0:4]

        print("Length of x_train 0 files: {}".format(len(self.x_train_0_files)))
        print("Length of x_train 1 files: {}".format(len(self.x_train_1_files)))
        print("Length of y_train files: {}".format(len(self.y_train_files)))

        main_list = np.setdiff1d(self.x_train_0_files, self.y_train_files)

        for item in main_list:

            if (item[0] != '.'):

                if item in self.x_train_0_files:

                    os.remove(os.path.join(self.in_path_0, item + ".png"))
                    os.remove(os.path.join(self.in_path_1, item + '.png'))
                
                elif item in y_train_files:

                    os.remove(os.path.join(out_path, item + ".npy"))

        # ^^ removing files present in one (x, y) and not other (x, y) to prevent dataset misalignment
        

        self.x_train_0_paths = []
        self.x_train_1_paths = []
        self.y_train_paths = []

        for item in os.listdir(in_path_0):

            if (item[-3: ] == 'png'):

                self.x_train_0_paths.append(item)

        for item in os.listdir(in_path_1):

            if (item[-3: ] == 'png'):

                self.x_train_1_paths.append(item)

        for item in os.listdir(out_path):

            if (item[-3: ] == 'npy'):

                self.y_train_paths.append(item)

        self.x_train_0_paths = sorted(self.x_train_0_paths)
        self.x_train_1_paths = sorted(self.x_train_1_paths)
        self.y_train_paths = sorted(self.y_train_paths)

        if (filelimiter < 1 or filelimiter> len(self.x_train_0_paths)):

            filelimiter = len(self.x_train_0_paths)

        for i in range(0, filelimiter):

            pt_in_0 = os.path.join(in_path_0, self.x_train_0_paths[i])
            pt_in_1 = os.path.join(in_path_1, self.x_train_1_paths[i])
            pt_out = os.path.join(out_path, self.y_train_paths[i])

            self.x_train_0.append(cv.imread(pt_in_0, cv.IMREAD_GRAYSCALE))
            self.x_train_1.append(cv.imread(pt_in_1, cv.IMREAD_GRAYSCALE))
            self.y_train.append(np.load(pt_out))

        self.x_train_0 = np.array(self.x_train_0).astype('float64')
        self.x_train_0 /= np.amax(self.x_train_0)

        self.x_train_1 = np.array(self.x_train_1).astype('float64')
        self.x_train_1 /= np.amax(self.x_train_1)
        
        for i in range(0, len(self.y_train)):

            item = self.y_train[i]
            item = np.clip(item, -1000, 3000)
            print("Item {} with max: {} and min: {}".format(i, np.amax(self.y_train[i]), np.amin(self.y_train[i])))
            normalized = normalize(item)
            self.y_train[i] = normalized
            print("Normalized item {} with max: {} and min: {}".format(i, np.amax(self.y_train[i]), np.amin(self.y_train[i])))
        
        self.y_train = np.array(self.y_train).astype('float64')


    def view_pair(self, index):

        f = plt.figure()

        f.add_subplot(1, 3, 1)
        plt.imshow(self.x_train_0[index], cmap = 'gray')
        plt.title('Theta = 0 of {}'.format(index))

        f.add_subplot(1, 3, 2)
        plt.imshow(self.x_train_1[index], cmap = 'gray')
        plt.title('Theta = 1 of {}'.format(index))

        f.add_subplot(1, 3, 3)
        plt.imshow(self.y_train[index][64], cmap = 'gray')
        plt.title("Tomography of {}".format(index))
        plt.show(block = True)

    def view_animation(self, index):

        if (self.y_train):

            array = self.y_train[index]

            ims = []

            for im in array:

                gg = plt.imshow(im, cmap = 'gray')
                ims.append([gg])

            fig = plt.figure()
            ani = animation.ArtistAnimation(fig, ims, interval = 50, blit = True)
            plt.title(index)
            plt.show()

        else:

            print("Error: dataset has not been initialized yet")

    def prepare(self):
        
        self.x_train_0 = np.reshape(self.x_train_0, (len(self.x_train_0), 1024, 1024, 1))
        self.x_train_1 = np.reshape(self.x_train_1, (len(self.x_train_1), 1024, 1024, 1))
        self.y_train = np.reshape(self.y_train, (len(self.y_train), 128, 128, 128, 1))
        print("Shape of X0: {}".format(self.x_train_0.shape))
        print("Shape of X1: {}".format(self.x_train_1.shape))
        print("Shape of Y: {}".format(self.y_train.shape))

class PairedDatasetQuad: #dataset for double view neural network model

    def __init__(self, in_path_0, in_path_1, in_path_2, in_path_3, out_path, filelimiter): #if filelimiter is -1 or > 943, unlimited.
        
        self.in_path_0 = in_path_0
        self.in_path_1 = in_path_1
        self.in_path_2 = in_path_2
        self.in_path_3 = in_path_3
        self.out_path = out_path

        self.x_train_0 = []
        self.x_train_1 = []
        self.x_train_2 = []
        self.x_train_3 = []
        self.y_train = []

        self.x_train_0_files = os.listdir(in_path_0)
        self.x_train_1_files = os.listdir(in_path_1)
        self.x_train_2_files = os.listdir(in_path_2)
        self.x_train_3_files = os.listdir(in_path_3)
        self.y_train_files = os.listdir(out_path)

        for i in range(0, len(self.x_train_0_files)):

            self.x_train_0_files[i] = self.x_train_0_files[i][0:4]

        for i in range(0, len(self.x_train_1_files)):

            self.x_train_1_files[i] = self.x_train_1_files[i][0:4]

        for i in range(0, len(self.x_train_2_files)):

            self.x_train_2_files[i] = self.x_train_2_files[i][0:4]

        for i in range(0, len(self.x_train_3_files)):

            self.x_train_3_files[i] = self.x_train_3_files[i][0:4]

        for i in range(0, len(self.y_train_files)):

            self.y_train_files[i] = self.y_train_files[i][0:4]

        print("Length of x_train 0 files: {}".format(len(self.x_train_0_files)))
        print("Length of x_train 1 files: {}".format(len(self.x_train_1_files)))
        print("Length of x_train 2 files: {}".format(len(self.x_train_2_files)))
        print("Length of x_train 3 files: {}".format(len(self.x_train_3_files)))
        print("Length of y_train files: {}".format(len(self.y_train_files)))

        main_list = np.setdiff1d(self.x_train_0_files, self.y_train_files) #assuming that corresponding 0, 1, 2, 3 views are inputted

        for item in main_list:

            if (item[0] != '.'):

                if item in self.x_train_0_files:

                    os.remove(os.path.join(self.in_path_0, item + ".png"))
                    os.remove(os.path.join(self.in_path_1, item + '.png'))
                    os.remove(os.path.join(self.in_path_2, item + '.png'))
                    os.remove(os.path.join(self.in_path_3, item + '.png'))
                
                elif item in y_train_files:

                    os.remove(os.path.join(out_path, item + ".npy"))

        # ^^ removing files present in one (x, y) and not other (x, y) to prevent dataset misalignment
        

        self.x_train_0_paths = []
        self.x_train_1_paths = []
        self.x_train_2_paths = []
        self.x_train_3_paths = []
        self.y_train_paths = []

        for item in os.listdir(self.in_path_0):

            if (item[-3: ] == 'png'):

                self.x_train_0_paths.append(item)

        for item in os.listdir(self.in_path_1):

            if (item[-3: ] == 'png'):

                self.x_train_1_paths.append(item)

        for item in os.listdir(self.in_path_2):

            if (item[-3:] == 'png'):

                self.x_train_2_paths.append(item)

        for item in os.listdir(self.in_path_3):

            if (item[-3:] == 'png'):

                self.x_train_3_paths.append(item)

        for item in os.listdir(out_path):

            if (item[-3: ] == 'npy'):

                self.y_train_paths.append(item)

        self.x_train_0_paths = sorted(self.x_train_0_paths)
        self.x_train_1_paths = sorted(self.x_train_1_paths)
        self.x_train_2_paths = sorted(self.x_train_2_paths)
        self.x_train_3_paths = sorted(self.x_train_3_paths)
        self.y_train_paths = sorted(self.y_train_paths)

        if (filelimiter < 1 or filelimiter> len(self.x_train_0_paths)):

            filelimiter = len(self.x_train_0_paths)

        for i in range(0, filelimiter):

            pt_in_0 = os.path.join(in_path_0, self.x_train_0_paths[i])
            pt_in_1 = os.path.join(in_path_1, self.x_train_1_paths[i])
            pt_in_2 = os.path.join(in_path_2, self.x_train_2_paths[i])
            pt_in_3 = os.path.join(in_path_3, self.x_train_3_paths[i])

            pt_out = os.path.join(out_path, self.y_train_paths[i])

            self.x_train_0.append(cv.imread(pt_in_0, cv.IMREAD_GRAYSCALE))
            self.x_train_1.append(cv.imread(pt_in_1, cv.IMREAD_GRAYSCALE))
            self.x_train_2.append(cv.imread(pt_in_2, cv.IMREAD_GRAYSCALE))
            self.x_train_3.append(cv.imread(pt_in_3, cv.IMREAD_GRAYSCALE))
            self.y_train.append(np.load(pt_out))

        self.x_train_0 = np.array(self.x_train_0).astype('float64')
        self.x_train_0 /= np.amax(self.x_train_0)

        self.x_train_1 = np.array(self.x_train_1).astype('float64')
        self.x_train_1 /= np.amax(self.x_train_1)

        self.x_train_2 = np.array(self.x_train_2).astype('float64')
        self.x_train_2 /= np.amax(self.x_train_2)

        self.x_train_3 = np.array(self.x_train_3).astype('float64')
        self.x_train_3 /= np.amax(self.x_train_3)
        
        for i in range(0, len(self.y_train)):

            item = self.y_train[i]
            item = np.clip(item, -1000, 3000)
            print("Item {} with max: {} and min: {}".format(i, np.amax(self.y_train[i]), np.amin(self.y_train[i])))
            normalized = normalize(item)
            self.y_train[i] = normalized
            print("Normalized item {} with max: {} and min: {}".format(i, np.amax(self.y_train[i]), np.amin(self.y_train[i])))
        
        self.y_train = np.array(self.y_train).astype('float64')


    def view_pair(self, index):

        f = plt.figure()

        f.add_subplot(1, 5, 1)
        plt.imshow(self.x_train_0[index], cmap = 'gray')
        plt.title('Theta 0 of {}'.format(index))

        f.add_subplot(1, 5, 2)
        plt.imshow(self.x_train_1[index], cmap = 'gray')
        plt.title('Theta 1 of {}'.format(index))

        f.add_subplot(1, 5, 3)
        plt.imshow(self.x_train_2[index], cmap = 'gray')
        plt.title('Theta 2 of {}'.format(index))

        f.add_subplot(1, 5, 4)
        plt.imshow(self.x_train_3[index], cmap = 'gray')
        plt.title('Theta 3 of {}'.format(index))

        f.add_subplot(1, 5, 5)
        plt.imshow(self.y_train[index][64], cmap = 'gray')
        plt.title("Tomography of {}".format(index))
        plt.show(block = True)

    def view_animation(self, index):

        if (self.y_train):

            array = self.y_train[index]

            ims = []

            for im in array:

                gg = plt.imshow(im, cmap = 'gray')
                ims.append([gg])

            fig = plt.figure()
            ani = animation.ArtistAnimation(fig, ims, interval = 50, blit = True)
            plt.title(index)
            plt.show()

        else:

            print("Error: dataset has not been initialized yet")

    def prepare(self):
        
        self.x_train_0 = np.reshape(self.x_train_0, (len(self.x_train_0), 1024, 1024, 1))
        self.x_train_1 = np.reshape(self.x_train_1, (len(self.x_train_1), 1024, 1024, 1))
        self.x_train_2 = np.reshape(self.x_train_2, (len(self.x_train_2), 1024, 1024, 1))
        self.x_train_3 = np.reshape(self.x_train_3, (len(selx.x_train_3), 1024, 1024, 1))
        self.y_train = np.reshape(self.y_train, (len(self.y_train), 128, 128, 128, 1))
        print("Shape of X0: {}".format(self.x_train_0.shape))
        print("Shape of X1: {}".format(self.x_train_1.shape))
        print("Shape of X2: {}".format(self.x_train_2.shape))
        print("Shape of X3: {}".format(self.x_train_3.shape))
        print("Shape of Y: {}".format(self.y_train.shape))
