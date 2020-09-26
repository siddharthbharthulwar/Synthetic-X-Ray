import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import os
import cv2 as cv
from random import randint
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D, Reshape, Flatten, Dense
from keras.layers.merge import concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.optimizers import Adam
from keras.initializers import RandomNormal

def normalize(array):

    normalized = (array - np.amin(array)) / (np.amax(array) - np.amin(array))

    return normalized

class GANDatasetSingle:

    def __init__(self, in_path, out_path, filelimiter):
        
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

    def define_discriminator(self):

        input_img_1 = Input(shape=(128, 128, 128, 1))    # adapt this if using 'channels_first' image data format
        input_img_2 = Input(shape = (1024, 1024, 1))

        x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(input_img_1)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Conv3D(1024, (3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Conv3D(2048, (3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Reshape((2, 2, 4096))(x)

        y = Conv2D(32, (3, 3), activation = "relu", padding = 'same')(input_img_2)
        y = MaxPooling2D((2, 2), padding = 'same')(y)
        y = Conv2D(64, (3, 3), activation = "relu", padding = 'same')(y)
        y = MaxPooling2D((2, 2), padding = 'same')(y)
        y = Conv2D(128, (3, 3), activation = "relu", padding = 'same')(y)
        y = MaxPooling2D((2, 2), padding = 'same')(y)
        y = Conv2D(256, (3, 3), activation = "relu", padding = 'same')(y)
        y = MaxPooling2D((2, 2), padding = 'same')(y)
        y = Conv2D(512, (3, 3), activation = "relu", padding = 'same')(y)
        y = MaxPooling2D((2, 2), padding = 'same')(y)
        y = Conv2D(512, (3, 3), activation = "relu", padding = 'same')(y)
        y = MaxPooling2D((2, 2), padding = 'same')(y)
        y = Conv2D(1024, (3, 3), activation = "relu", padding = 'same')(y)
        y = MaxPooling2D((2, 2), padding = 'same')(y)
        y = Conv2D(2048, (3, 3), activation = "relu", padding = 'same')(y)
        y = MaxPooling2D((2, 2), padding = 'same')(y)
        y = Conv2D(4096, (3, 3), activation = "relu", padding = 'same')(y)
        y = MaxPooling2D((2, 2), padding = 'same')(y)

        encoded = concatenate([x, y])
        encoded = Reshape(target_shape=(32, 32, 32))(encoded)
        encoded = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(encoded)
        encoded = MaxPooling2D((2, 2), padding = 'same')(encoded)
        encoded = Conv2D(1, (1, 1), activation = 'relu', padding = 'same')(encoded)
        
        discriminator = Model([input_img_2, input_img_1], encoded)
        discriminator.summary()

        opt = Adam(learning_rate=0.0002, beta_1 = 0.5)
        discriminator.compile(loss = 'binary_crossentropy', optimizer = opt, loss_weights = [0.5])
        self.discriminator = discriminator
        print("Discriminator created")

    def define_generator(self):

        input_img = Input((1024, 1024, 1))

        init = RandomNormal(stddev = 0.02)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(2048, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(4096, (3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        encoded = Reshape((4, 4, 4, 1024))(encoded)

        x = Conv3D(512, (3, 3, 3), activation = 'relu', padding = 'same')(encoded)
        x = UpSampling3D((2, 2, 2))(x)

        x = Conv3D(256, (3, 3, 3), activation = 'relu', padding = 'same')(x)
        x = UpSampling3D((2, 2, 2))(x)

        x = Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same')(x)
        x = UpSampling3D((2, 2, 2))(x)

        x = Conv3D(64, (3, 3, 3), activation = 'relu', padding = 'same')(x)
        x = UpSampling3D((2, 2, 2))(x)

        x = Conv3D(32, (3, 3, 3), activation = 'relu', padding = 'same')(x)
        x = UpSampling3D((2, 2, 2))(x)

        decoded = Conv3D(1, (1, 1, 1), activation = 'relu', padding = 'same')(x)

        generator = Model(input_img, decoded)
        self.generator = generator
        print("Generator created")

    def make_gan(self):

        self.discriminator.trainable = False

        in_src = Input((1024, 1024, 1))

        gen_out = self.generator(in_src)
        dis_out = self.discriminator([in_src, gen_out])
        model = Model(in_src, [dis_out, gen_out])

        opt = Adam(lr = 0.00015, beta_1 = 0.5)
        model.compile(loss = ['binary_crossentropy', 'mae'], optimizer = opt, loss_weights= [1, 100])
        self.gan = model

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

    def generate_real_samples(self, n_samples, patch_shape):

        ix = np.random.randint(0, self.x_train.shape[0], n_samples)

        X1, X2 = self.x_train[ix], self.y_train[ix]
        
        y = np.ones((n_samples, patch_shape, patch_shape, 1))
        return [X1, X2], y

    def generate_fake_samples(self, samples, patch_shape):

        X = self.generator.predict(samples)
        y = np.zeros((len(X), patch_shape, patch_shape, 1))
        return X, y

    def train(self, n_epochs = 100, n_batch = 1, n_patch = 16):

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
            X_fakeB, y_fake = self.generate_fake_samples(X_realA, n_patch)
            d_loss1 = self.discriminator.train_on_batch([X_realA, X_realB], y_real)
            d_loss2 = self.discriminator.train_on_batch([X_realA, X_fakeB], y_fake)

            g_loss, _, _ = self.gan.train_on_batch(X_realA, [y_real, X_realB])

            self.d1_losses.append(d_loss1)
            self.d2_losses.append(d_loss2)
            self.g_losses.append(g_loss)
            
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))

            if (i % len(self.x_train) == 0):

                self.d1_epoch_losses.append(d_loss1)
                self.d2_epoch_losses.append(d_loss2)
                self.g_epoch_losses.append(g_loss)

                temp_in = np.reshape(self.x_train[555], (1, 1024, 1024, 1))
                temp_out = self.generator.predict(temp_in)
                temp_out = np.reshape(temp_out, (128, 128, 128))
                real_out = np.reshape(self.y_train[555], (128, 128, 128))

                f = plt.figure()
                f.add_subplot(1, 2, 1)
                plt.imshow(real_out[64], cmap = 'gray', vmin = 0, vmax = 1)

                f.add_subplot(1, 2, 2)
                plt.imshow(temp_out[64], cmap = 'gray', vmin = 0, vmax = 1)

                plt.suptitle("Epoch: {}".format(int(i / len(self.x_train))))

                plt.show(block = True)

