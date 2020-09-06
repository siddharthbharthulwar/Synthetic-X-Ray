import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv

class SinglePair:

    def __init__(self, xr_path, ct_path):
        
        self.xr_path = xr_path
        self.ct_path = ct_path

        print("xr: {}".format(self.xr_path))
        print("ct: {}".format(self.ct_path))


class GANDatasetSingle:

    def __init__(self, in_path, out_path):
        
        self.in_path = in_path
        self.out_path = out_path

        self.x_train_files = os.listdir(self.in_path)
        self.y_train_files = os.listdir(self.out_path)

        for i in range(0, len(self.x_train_files)):

            self.x_train_files[i] = self.x_train_files[i][0: 4]
        
        for i in range(0, len(self.y_train_files)):

            self.y_train_files[i] = self.y_train_files[i][0: 4]

        print("Length of x_train files: {}".format(len(self.x_train_files)))
        print("Length of y_train files: {}".format(len(self.y_train_files)))

        main_list = np.setdiff1d(self.x_train_files, self.y_train_files)

        for item in main_list:

            if (item[0] != '.'):

                if item in self.x_train_files:

                    os.remove(os.path.join(self.in_path, item + '.png'))

                elif item in self.y_train_files:

                    os.remove(os.path.join(self.out_path, item + '.npy'))

        self.x_train_paths = []
        self.y_train_paths = []

        for item in os.listdir(self.in_path):

            if (item[-3: ] == 'png'):

                self.x_train_paths.append(item)

        for item in os.listdir(self.out_path):

            if (item[-3: ] == 'npy'):

                self.y_train_paths.append(item)

        self.x_train_paths = sorted(self.x_train_paths)
        self.y_train_paths = sorted(self.y_train_paths)
        print(self.x_train_paths)
        print(self.y_train_paths)
        self.pairs = []
        
        for i in range(0, 300):
            print(self.x_train_paths[i])
            print(self.y_train_paths[i])
            #sl = SinglePair(os.path.join(self.in_path, self.x_train_paths[i]), os.path.join(self.out_path, self.y_train_paths[i]))
        

g = GANDatasetSingle('Data/In/0', 'Data/Out_New')