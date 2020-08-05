import numpy as np
import os
from scipy.ndimage import zoom

root = r"Data\Out"

for item in os.listdir(root):

    array = np.load(os.path.join(root, item))
    reshaped = zoom(array, (0.5, 0.25, 0.25))
    np.save(os.path.join(r"Data\Out_New", item), reshaped)
    print(item)