import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import os
import pydicom as dicom
import scipy.ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

from skimage import measure, morphology


# Load the scans in given folder path
def load_scan(path):

    slices = []

    for s in os.listdir(path):

        if (s[-3: ] == 'dcm'):

            slices.append(dicom.read_file(path + '/' + s))

    #slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
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


def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    print("Calculating Surface Mesh")
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold)
    print(verts.shape, faces.shape)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image


def returnNumSlices(pathStr):

    first_patient = load_scan(pathStr)
    first_patient_pixels = get_pixels_hu(first_patient)

    return (first_patient_pixels.shape[0])

def viewVolume(pathStr):

    patients = os.listdir(pathStr)
    patients.sort()

    first_patient = load_scan(pathStr)
    first_patient_pixels = get_pixels_hu(first_patient)
    plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()


    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
    print("Shape before resampling\t", first_patient_pixels.shape)
    print("Shape after resampling\t", pix_resampled.shape)

    
    segmented_lungs = segment_lung_mask(pix_resampled, False)
    segmented_lungs_fill = segment_lung_mask(pix_resampled, True)

    #plot_3d(segmented_lungs_fill - segmented_lungs, 0)
    arr = segmented_lungs - segmented_lungs_fill

    ims = []

    for im in arr:

        gg = plt.imshow(im, cmap = 'gray')
        ims.append([gg])

    fig = plt.figure()

    ani = animation.ArtistAnimation(fig, ims, interval = 50, blit = True, repeat_delay = 2000)

    plt.show()


def view3DSegmentation(image):

    segmented_lungs = segment_lung_mask(image, False)
    segmented_lungs_fill = segment_lung_mask(image, True)

    arr = segmented_lungs - segmented_lungs_fill

    ims = []

    for im in arr:

        gg = plt.imshow(im, cmap = 'gray')
        ims.append([gg])

    fig = plt.figure()

    ani = animation.ArtistAnimation(fig, ims, interval = 50, blit = True, repeat_delay = 500)

    plt.show()

'''
array = np.load(r'Data\Out_New\0002.npy')
segmented_lungs = segment_lung_mask(array, False)
segmented_lungs_fill = segment_lung_mask(array, True)

arr = segmented_lungs - segmented_lungs_fill

plot_3d(arr, 0)

'''