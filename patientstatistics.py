import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom as dicom

class Patient:

    def __init__(self, dicom_object):
        
        self.AcquisitionDate = dicom_object.AcquisitionDate
        self.FocalSpots = dicom_object.FocalSpots
        self.Manufacturer = dicom_object.Manufacturer
        self.WindowCenter = dicom_object.WindowCenter
        self.WindowWidth = dicom_object.WindowWidth


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
                patients.append(Patient(sl))
                index +=1

    print("Patient: {}".format(str(patient_identifier_index)))
    patient_identifier_index +=1
    
print(len(patients))



