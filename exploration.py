import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import os
import pydicom as dicom
import scipy.ndimage
from os import listdir
from os.path import isfile, join

def load_scan(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def viewCoronalSlice(img, index):

    plt.imshow(img[index, :, :], cmap = 'gray')
    plt.show()

root = r"D:\Documents\School\2020-21\CT\LIDC-IDRI"
#imagine doing it the correct way lmao can't relate

def crawl(root):

    dirlist = os.listdir(root)

    for one in dirlist:

        dirlisttwo = os.listdir(os.path.join(root, one))

        for two in dirlisttwo:
            
            dirlistthree = os.listdir(os.path.join(root, one, two))


            for three in dirlistthree:
                
                dirlistfour = os.listdir(os.path.join(root, one, two, three))

                if (len(dirlistfour) > 170):

                    patient = load_scan(os.path.join(root, one, two, three))
                    hu = get_pixels_hu(patient)
                    #resampled, spacing = resample(hu, patient)

                    viewCoronalSlice(hu, 50)
crawl(root)