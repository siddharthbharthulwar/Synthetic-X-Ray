import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom as dicom

def dicom_stats(path_to_dicom):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    for sl in slices:

        print(type(sl))
#file for viewing patient statistics of LIDC-IDRI dataset

root = r"D:\Documents\School\2020-21\CT\LIDC-IDRI"
dataset_count = 0

num_slices = []

for path, subdirs, files in os.walk(root):
    counter = 0
    for name in files:
        if (name[-3: ] == 'dcm'):
            counter +=1
    if (counter > 50):
        
        print(path)
    
