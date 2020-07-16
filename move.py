import cv2 as cv
import os
import shutil

path = r"D:\Documents\School\2020-21\CT\LIDC-IDRI"


def crawl_move_singlevolume(path, num):

    start = r"Data\Out"

    for file in os.listdir(path):

        subpath = os.path.join(path, file)
        subsubpath = os.listdir(subpath)[0]
        num_dicoms = len(os.listdir(os.path.join(subpath, subsubpath)))

        if (num_dicoms > 100):
            
            source = os.path.join(subpath, subsubpath)
            destination = os.path.join(start, num)
            dest = shutil.move(source, destination)
            print(dest)

def crawl(root):

    patch = r"Data\In\0"


    for file in os.listdir(root):

        marker = ((file[-4:]) + '.png')
        if (marker in os.listdir(patch)):

            crawl_move_singlevolume(os.path.join(root, file), file[-4:])

crawl(path)