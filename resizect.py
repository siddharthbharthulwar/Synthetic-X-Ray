import numpy as np
import os
from scipy.ndimage import zoom

root = r"Data\Out"

for item in os.listdir(root):

    if item not in os.listdir(r'Data/Out_NN'):
        print(item)
        array = np.load(os.path.join(root, item))
        reshaped = zoom(array, (1, 0.5, 0.5))
        np.save(os.path.join(r"Data\Out_NN", item), reshaped)
    