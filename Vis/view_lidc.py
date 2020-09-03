import os
import matplotlib.pyplot as plt
import numpy 

counter = 0
root = 'Data/Out'

for file in os.listdir(root):

    if (counter < 8):

        arr = numpy.load(os.path.join(root, file))
        arr = numpy.clip(arr, -1000, 3000)
        plt.imshow(arr[128], cmap = 'gray')
        plt.show()
        counter +=1