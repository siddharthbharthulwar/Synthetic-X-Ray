import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import radon, iradon
from scipy.ndimage import zoom
import os
import pydicom as dicom
import scipy.ndimage
import numpy as np
import matplotlib.animation as animation


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

def radonTransformation(img, angle):

    img = zoom(img, 0.4)

    projections = radon(img, theta = [angle])
    return projections

def radonShow(img, angle):

    image = zoom(img, 0.4)

    plt.figure(figsize=(8, 8.5))

    plt.subplot(221)
    plt.title("Original");
    plt.imshow(image, cmap=plt.cm.Greys_r)

    plt.subplot(222)
    projections = radon(image, theta=[angle])
    plt.plot(projections);
    plt.title("Projections at\n0 degrees")
    plt.xlabel("Projection axis")
    plt.ylabel("Intensity")

    projections = radon(image)
    plt.subplot(223)
    plt.title("Radon transform\n(Sinogram)")
    plt.xlabel("Projection axis")
    plt.ylabel("Intensity")
    plt.imshow(projections)

    reconstruction = iradon(projections)
    plt.subplot(224)
    plt.title("Reconstruction\nfrom sinogram")
    plt.imshow(reconstruction, cmap=plt.cm.Greys_r)

    plt.subplots_adjust(hspace=0.4, wspace=0.5)
    plt.show()

def createXRay(volumePath, angle):

    b = get_pixels_hu(load_scan(volumePath))

    ls = []

    for sl in b:

        ls.append(radonTransformation(sl, angle))

    ls = np.array(ls)

    return np.rot90(np.squeeze(ls), 2)

def volumeRotationAnimation(volumePath):

    b = get_pixels_hu(load_scan(volumePath))
    ls = []

    for i in range(0, 359):

        print(i)
        gg = plt.imshow(createXRay(volumePath, i), cmap = 'gray')
        ls.append([gg])

    fig = plt.figure()

    ani = animation.ArtistAnimation(fig, ls, interval = 50, blit = True, repeat_delay = 3000)

    plt.show()


#volumeRotationAnimation('chestCT0/volume2/')

