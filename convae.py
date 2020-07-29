import numpy as np
import matplotlib.pyplot as plt

### Autoencoder ###
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

import cv2 as cv
import os

view0 = []
view1 = []
view2 = []
view3 = []

counter = 0

for a in os.listdir('Data/In/0'):

    temp_img = cv.imread(os.path.join('Data/In/0', a), cv.IMREAD_GRAYSCALE)
    view0.append(temp_img)

for b in os.listdir('Data/In/0'):

    temp_img = cv.imread(os.path.join('Data/In/1', b), cv.IMREAD_GRAYSCALE)
    view1.append(temp_img)

for c in os.listdir('Data/In/0'):

    temp_img = cv.imread(os.path.join('Data/In/2', c), cv.IMREAD_GRAYSCALE)
    view2.append(temp_img)

for d in os.listdir('Data/In/0'):

    temp_img = cv.imread(os.path.join('Data/In/3', d), cv.IMREAD_GRAYSCALE)
    view3.append(temp_img)

view0 = np.array(view0)
view1 = np.array(view1)
view2 = np.array(view2)
view3 = np.array(view3)

view0 = view0.astype('float32') / 255
view1 = view1.astype('float32') / 255
view2 = view2.astype('float32') / 255
view3 = view3.astype('float32') / 255

view0 = np.reshape(view0, (len(view0), 1024, 1024, 1)) 
view1 = np.reshape(view1, (len(view1), 1024, 1024, 1))
view2 = np.reshape(view2, (len(view2), 1024, 1024, 1)) 
view3 = np.reshape(view3, (len(view3), 1024, 1024, 1))           

input_img = Input(shape=(1024, 1024, 1))    # adapt this if using 'channels_first' image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

encoder = Model(input_img, encoded)


# at this point the representation is (4, 4, 8), i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()

x_train = np.reshape(view0, (len(view0), 1024, 1024, 1))    # adapt this if using 'channels_first' image data format
y_train = np.reshape(view1, (len(view1), 1024, 1024, 1))    # adapt this if using 'channels_first' image data format

# open a terminal and start TensorBoard to read logs in the autoencoder subdirectory
# tensorboard --logdir=autoencoder
autoencoder.compile(optimizer='adam', loss = 'mse')
autoencoder.fit(x_train, y_train, epochs=41, batch_size=4, shuffle=True)
print("model has been fit")
# take a look at the reconstructed digits
print("past showing and saving")
autoencoder.save('autoencoder.h5')
print("autoencoder has been saved")

del autoencoder

