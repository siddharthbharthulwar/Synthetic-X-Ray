import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import shutil

def isFlipped(path):

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    fig = plt.figure()
    plt.imshow(img, cmap = 'gray')
    plt.title(path)

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    val = input("F: Flip")
    print(val)

    if (val == 'f'):

        return True

    else:

        return False

def testslice(array):

    plt.imshow(array[:, 100, :], cmap = 'gray')
    plt.show()

def flip_ct(array):

    flipped = np.rot90(array, k = 2, axes = (2, 0))
    return flipped

def flip_xr(array):

    flipped = np.rot90(array, k = 2)
    return flipped

def move(path, outpath):

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    cv.imwrite(outpath, img)

root = 'D:\Documents\GitHub\GitHub\Synthetic-X-Ray\Matlab_SXR\CXR'

flippedlist = []
normal = []

for path, subdirs, files in os.walk(root):
    for name in subdirs:
        subpath = os.path.join(root ,name)
        for spath, ssubdirs, sfiles in os.walk(subpath):

            flipped = False
            for bname in sfiles:

                if (bname == '0.png'):

                    flipped = isFlipped(os.path.join(subpath, bname))

            for sname in sfiles:

                if (sname == '0.png'):

                    finalpath = os.path.join('Data/In/0/', name + '.png')

                    if (flipped):
                        
                        ff1 = cv.imread(os.path.join(subpath, sname), cv.IMREAD_GRAYSCALE)
                        ff1 = cv.rotate(ff1, cv.ROTATE_180)
                        cv.imwrite(finalpath, ff1)
                
                        flippedlist.append(subpath)

                        ct = np.load(os.path.join('Data\Out', name + '.npy'))
                        ct_flipped = flip_ct(ct)

                        np.save(os.path.join('Data\Out', name), ct_flipped)

                    else:

                        normal.append(subpath)
                        shutil.copy(os.path.join(subpath, sname), finalpath)

                elif (sname == '1.png'):

                    
                    if (flipped):

                        finalpath = (os.path.join('Data/In/3/', name + '.png'))
                        ff1 = cv.imread(os.path.join(subpath, sname), cv.IMREAD_GRAYSCALE)
                        ff1 = cv.rotate(ff1, cv.ROTATE_180)
                        cv.imwrite(finalpath, ff1)

                    else:

                        finalpath = (os.path.join('Data/In/1/', name + '.png'))
                        shutil.copy(os.path.join(subpath, sname), finalpath)

                elif (sname == '2.png'):

                    finalpath = (os.path.join('Data/In/2/', name + '.png'))

                    if (flipped):

                        ff1 = cv.imread(os.path.join(subpath, sname), cv.IMREAD_GRAYSCALE)
                        ff1 = cv.rotate(ff1, cv.ROTATE_180)
                        cv.imwrite(finalpath, ff1)

                    else:

                        shutil.copy(os.path.join(subpath, sname), finalpath)

                elif (sname == '3.png'):

                
                    if (flipped):
                        
                        finalpath = (os.path.join('Data/In/1/', name + '.png'))
                        ff1 = cv.imread(os.path.join(subpath, sname), cv.IMREAD_GRAYSCALE)
                        ff1 = cv.rotate(ff1, cv.ROTATE_180)
                        cv.imwrite(finalpath, ff1)
                    
                    else:
                        
                        finalpath = (os.path.join('Data/In/3/', name + '.png'))
                        shutil.copy(os.path.join(subpath, sname), finalpath)

                    #flip180(os.path.join(subpath, name, "placeholder"), False)


#write crawler to manually inspect all of them 
