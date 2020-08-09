import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import os
import cv2 as cv

class PairedDataset:

    def __init__(self, in_path, out_path):
        
        self.in_path = in_path
        self.out_path = out_path

        self.x_train = []
        self.y_train = []

        self.x_train_files = os.listdir(in_path)
        self.y_train_files = os.listdir(out_path)

        for i in range(0, len(x_train_files)):

            x_train_files[i] = x_train_file[i][0:4]

        for i in range(0, len(y_train_files)):

            y_train_files[i] = y_train_files[i][0:4]

        print("Length of x_train files: {}".format(len(x_train_files)))
        print("Length of y_train files: {}".format(len(y_train_files)))

        main_list = np.setdiff1d(x_train_files, y_train_files)

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

        for i in range(0, len(self.x_train_paths)):

            pt_in = os.path.join(in_path, self.x_train_paths[i])
            pt_out = os.path.join(out_path, self.y_train_paths[i])

            self.x_train.append(cv.imread(pt_in, cv.IMREAD_GRAYSCALE))
            self.y_train.append(np.load(pt_out))

        self.x_train = np.array(self.x_train).astype('float64')
        self.x_train /= np.amax(self.x_train)


    def view_pair(self, index):

        print("not done")

    def view_animation(self, index):

        if (self.y_train):

            array = self.y_train[index]

            ims = []

            for im in arr:

                gg = plt.imshow(im, cmap = 'gray')
                ims.append([gg])

            fig = plt.figure()
            ani = animation.ArtistAnimation(fig, ims, interval = 50, blit = True)
            plt.title(index)
            plt.show()

        else:

            print("Error: dataset has not been initialized yet")



    