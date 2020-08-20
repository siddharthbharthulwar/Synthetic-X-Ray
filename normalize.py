import numpy as np
import os
import matplotlib.pyplot as plt

root = r'Data\Out_New'

for path in os.listdir(root):

    arr = np.load(os.path.join(root, path))
    f = plt.figure(figsize=(15, 8))

    f.add_subplot(1, 2, 1)
    plt.imshow(arr[64], cmap = 'gray')
    
    f.add_subplot(1, 2, 2)
    plt.hist(arr.ravel(), bins = 50)
    plt.suptitle(path)
    f.savefig(os.path.join('Data\Histograms', path[0:4]))
    plt.close(f)
    print(path)
