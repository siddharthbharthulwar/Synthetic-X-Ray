import cv2
import numpy as np
import math
import pydicom

def DICOMwaterequivalent(dicom_filename, threshold):

    # load DICOM image
    dicom_pydicom  = pydicom.read_file(dicom_filename)
    dicom_img = dicom_pydicom.pixel_array # dicom pixel values as 2D numpy pixel array
    dicom_img = dicom_img - 1000.0 # remap scale 0:... to HU -1000:...

    # determine pixel area in mm²/px²
    scale = dicom_pydicom.PixelSpacing[0] * dicom_pydicom.PixelSpacing[1]

    # map ww/wl for contour detection (filter_img)
    remap = lambda t: 255.0 * (1.0 * t - (threshold - 0.5))
    filter_img = np.array([remap(row) for row in dicom_img])
    filter_img = np.clip(filter_img, 0, 255)
    filter_img = filter_img.astype(np.uint8)

    # find contours, without hierarchical relationships
    ret,thresh = cv2.threshold(filter_img, 127, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # calculate area and equivalent circle diameter for the largest contour (assumed to be the patient without table or clothing)
    contour = max(contours, key=lambda a: cv2.contourArea(a))
    area = cv2.contourArea(contour) * scale
    equiv_circle_diam = 2.0*math.sqrt(area/math.pi)

    hull = cv2.convexHull(contour)
    hullarea = cv2.contourArea(hull) * scale
    hullequiv = 2.0*math.sqrt(hullarea/math.pi)

    # create mask of largest contour
    mask_img = np.zeros((dicom_img.shape), np.uint8)
    cv2.drawContours(mask_img,[contour],0,255,-1)

    # calculate mean HU of mask area
    roi_mean_hu = cv2.mean(dicom_img, mask=mask_img)[0]

    # calculate water equivalent area (Aw) and water equivalent circle diameter (Dw)
    water_equiv_area = 0.001 * roi_mean_hu * area + area
    water_equiv_circle_diam = 2.0 * math.sqrt(water_equiv_area/math.pi)

    return( area, equiv_circle_diam, water_equiv_area, water_equiv_circle_diam, hullarea, hullequiv)


s = DICOMwaterequivalent(r"D:\Documents\School\2020-21\CT\LIDC-IDRI\LIDC-IDRI-0008\01-01-2000-30141\3000549.000000-21954\1-104.dcm", -250)
print(s)