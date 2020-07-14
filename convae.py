import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Autoencoder ###
import tensorflow as tf
import tensorflow.keras

from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, model_from_json

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Input

from tensorflow.keras.datasets import mnist

import cv2 as cv
import os

view0 = []
view1 = []
view2 = []
view3 = []

counter = 0

for x in os.listdir('CXR'):

    for y in os.listdir(os.path.join('CXR', x)):


        path = os.path.join('CXR', x, y)
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)

        if (y == '0.png'):

            view0.append(img)

        elif (y == '1.png'):

            view1.append(img)

        elif (y == '2.png'):

            view2.append(img)

        elif (y == '3.png'):

            view3.append(img)

    print(counter)
    counter +=1

