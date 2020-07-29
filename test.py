import os
import numpy as np
import matplotlib.pyplot as plt 

root = r"Data\Out"

for file in os.listdir(root):

    array = np.load(os.path.join(root, file))
    plt.imshow(array[150, :, :], cmap = 'gray')
    plt.show()

