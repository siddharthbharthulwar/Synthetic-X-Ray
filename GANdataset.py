import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import os
import cv2 as cv
from random import randint

def normalize(array):

    normalized = (array - np.amin(array)) / (np.amax(array) - np.amin(array))

    return normalized

class GANDatasetSingle:

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

        print("LENGTH OF MAIN LIST: {}".format(len(main_list)))

        '''

        for item in main_list:

            if (item[0] != '.'):

                if item in x_train_files:

                    os.remove(os.path.join(in_path, item + ".png"))
                
                elif item in y_train_files:

                    os.remove(os.path.join(out_path, item + ".npy"))

        '''

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

    def generate_real_samples(self, n_samples, patch_shape):

        ix = np.random.randint(0, self.x_train.shape[0], n_samples)

        X1, X2 = self.x_train[ix], self.y_train[ix]
        
        y = np.ones((n_samples, patch_shape, patch_shape, 1))
        return [X1, X2], y

    def generate_fake_samples(self, g_model, samples, patch_shape):

        X = g_model.predict(samples)
        y = np.zeros((len(X), patch_shape, patch_shape, 1))
        return X, y

    def train(self, d_model, g_model, gan_model, n_epochs = 100, n_batch = 1, n_patch = 16):

        self.d1_losses = []
        self.d2_losses = []
        self.g_losses = []

        self.d1_epoch_losses = []
        self.d2_epoch_losses = []
        self.g_epoch_losses = []

        bat_per_epo = int(len(self.x_train) / n_batch)

        n_steps = bat_per_epo * n_epochs

        for i in range(n_steps):

            [X_realA, X_realB], y_real = self.generate_real_samples(n_batch, n_patch)
            X_fakeB, y_fake = self.generate_fake_samples(g_model, X_realA, n_patch)
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

            self.d1_losses.append(d_loss1)
            self.d2_losses.append(d_loss2)
            self.g_losses.append(g_loss)
            
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))

            if (i % len(self.x_train) == 0):

                self.d1_epoch_losses.append(d_loss1)
                self.d2_epoch_losses.append(d_loss2)
                self.g_epoch_losses.append(g_loss)

                temp_in = np.reshape(self.x_train[555], (1, 1024, 1024, 1))
                temp_out = g_model.predict(temp_in)
                temp_out = np.reshape(temp_out, (128, 128, 128))
                real_out = np.reshape(self.y_train[555], (128, 128))

                f = plt.figure()
                f.add_subplot(1, 2, 1)
                plt.imshow(real_out[64], cmap = 'gray')

                plt.add_subplot(1, 2, 2)
                plt.imshow(temp_out[64], cmap = 'gray')

                plt.suptitle("Epoch: {}".format(int(i / len(self.x_train))))

                plt.show(block = True)


