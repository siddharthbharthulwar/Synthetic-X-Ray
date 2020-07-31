from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D, Reshape, Flatten
from keras.models import Model
from keras.layers.merge import concatenate
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras import backend as K
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

input_img = Input(shape=(1024, 1024, 1))    # adapt this if using 'channels_first' image data format
input_img_2 = Input(shape = (1024, 1024, 1))

x = Conv2D(4, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
encoded_1 = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(4, (3, 3), activation='relu', padding='same')(input_img_2)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
encoded_2 = MaxPooling2D((2, 2), padding='same')(x)

encoded = concatenate([encoded_1, encoded_2])

encoded = Reshape((8, 16, 16, 1))(encoded)

x = Conv3D(32, (3, 3, 3), activation="relu", padding="same")(encoded)
x = UpSampling3D((2, 2, 2))(x)
x = Conv3D(32, (3, 3, 3), activation="relu", padding="same")(x)
x = UpSampling3D((2, 2, 2))(x)
x = Conv3D(8, (3, 3, 3), activation="relu", padding="same")(x)
x = UpSampling3D((2, 2, 2))(x)
x = Conv3D(8, (3, 3, 3), activation="relu", padding="same")(x)
x = UpSampling3D((2, 2, 2))(x)
x = Conv3D(8, (3, 3, 3), activation="relu", padding="same")(x)
x = UpSampling3D((2, 2, 2))(x)
x = Conv3D(1, (3, 3, 3), activation="relu", padding="same")(x)
decoded = UpSampling3D((1, 1, 1))(x)

autoencoder = Model(inputs=[input_img, input_img_2], outputs = decoded)