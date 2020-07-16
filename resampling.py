import matplotlib.pyplot as plt
from skimage.io import imread
import scipy.ndimage
import numpy as np
import os
import pydicom as dicom
from numpy import asarray
from numpy import save


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

def resample(path):

    array = get_pixels_hu(load_scan(path))

    whole = np.zeros((256, 512, 512))

    dsfactor = [w/float(f) for w, f in zip(whole.shape, array.shape)]
    downed = scipy.ndimage.interpolation.zoom(array, zoom = dsfactor)

    return downed

def compare(path):

    array = get_pixels_hu(load_scan(path))

    whole = np.zeros((256, 512, 512))

    dsfactor = [w/float(f) for w,f in zip(whole.shape, array.shape)]
    downed = scipy.ndimage.interpolation.zoom(array, zoom=dsfactor)

    plt.imshow(array[:, 200, :], cmap = 'gray')
    plt.title(array.shape)
    plt.show()


    plt.imshow(downed[:, 200, :], cmap = 'gray')
    plt.title(downed.shape)
    plt.show()



for file in os.listdir(r"Data\Out_Old"):

    resampled = resample(os.path.join(r"Data\Out_Old", file))
    np.save(os.path.join(r"Data\Out", file + ".npy"), resampled)
    print(file)

    