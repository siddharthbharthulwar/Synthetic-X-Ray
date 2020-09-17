import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom as dicom




#file for viewing patient statistics of LIDC-IDRI dataset

root = r"D:\Documents\School\2020-21\CT\LIDC-IDRI"
dataset_count = 0

num_slices = []
patients = []

patient_identifier_index = 0

for path, subdirs, files in os.walk(root):
    counter = 0
    for name in files:
        if (name[-3: ] == 'dcm'):
            counter +=1
    if (counter < 50):
        
        index = 0
        for item in os.listdir(path):

            if (item[-3:] == 'dcm') and (index < 1):
                
                sl = dicom.read_file(os.path.join(path, item))

                index +=1
                print(path)

    patient_identifier_index +=1
    


