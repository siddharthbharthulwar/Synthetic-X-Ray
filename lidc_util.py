#file for getting original lidc dicom files
import numpy as np
import os
from viewdicom import viewVolume, returnNumSlices
#from resampling import resample

'''
rt = r"D:\Documents\School\2020-21\CT\LIDC-IDRI"

def get_scan_dirs(root):

    print("not done yet")

root = r"D:\Documents\School\2020-21\CT\LIDC-IDRI"
dataset_count = 0

num_slices = []

for path, subdirs, files in os.walk(root):
    counter = 0
    for name in files:
        if (name[-3: ] == 'dcm'):
            counter +=1
    if (counter > 50):
        slicenum = returnNumSlices(path)
        print(path[51:55])
        dataset_count +=1
        print(slicenum)
        num_slices.append(slicenum)

print(dataset_count)

num_slices = np.array(num_slices)

np.save('num_slices.npy', num_slices)

        #resampled = resample(path)

        #np.save(os.path.join(r"Data\Out", file + ".npy"), resampled)

'''

nums = np.load('num_slices.npy')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set_style('darkgrid')
sns.distplot(nums, kde=False)
plt.show()    
