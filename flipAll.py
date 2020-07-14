import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def isFlipped2(path):

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    fig = plt.figure()
    plt.imshow(img, cmap = 'gray')
    plt.title(path)

    plt.draw()
    plt.waitforbuttonpress(0) # this will wait for indefinite time
    plt.close(fig)

    val = input("F for Flip, S for no Flip")

    print(val)

    if (val == 'f'):

        return 1

    elif (val == 'd'):

        return 3

    else:

        return 2

def flip180(path, bool, outpath):

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img = cv.rotate(img, cv.ROTATE_180)
    if (bool):
        cv.imwrite(path, img)
    else:

        plt.imshow(img, cmap = 'gray')
        plt.title(path)
        plt.show()

def move(path, outpath):

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    cv.imwrite(outpath, img)

root = 'CXR'

removed = []
flipped = []
normal = []

for path, subdirs, files in os.walk(root):
    for name in subdirs:
        subpath = os.path.join(root, name)
        for spath, ssubdirs, sfiles in os.walk(subpath):

            flipped = 0
            for bname in sfiles:

                if (bname == '0.png'):

                    flipped = isFlipped2(os.path.join(subpath, bname))

            for sname in sfiles:

                if (sname == '0.png'):

                    finalpath = os.path.join('Data/In/0/', name + '.png')

                    if (flipped == 1):
                        
                        flip180(os.path.join(subpath, sname), True, finalpath)
                        flipped.append(subpath)

                    elif (flipped == 3):

                        os.rmdir(subpath)
                        removed.append(subpath)

                    else:

                        normal.append(subpath)
                        move(os.path.join(subpath, sname), finalpath)

                elif (sname == '1.png'):

                    finalpath = (os.path.join('Data/In/1/', name + '.png'))
                    
                    if (flipped == 1):

                        flip180(os.path.join(subpath, sname), True, finalpath)

                    else:

                        move(os.path.join(subpath, sname), finalpath)

                elif (sname == '2.png'):

                    finalpath = (os.path.join('Data/In/2/', name + '.png'))

                    if (flipped == 1):

                        flip180(os.path.join(subpath, sname), True, finalpath)

                    else:

                        move(os.path.join(subpath, sname), finalpath)

                elif (sname == '3.png'):

                    finalpath = (os.path.join('Data/In/3/', name + '.png'))
                
                    if (flipped == 1):

                        flip180(os.path.join(subpath, sname), True, finalpath)
                    
                    else:

                        move(os.path.join(subpath, sname), finalpath)

                    #flip180(os.path.join(subpath, name, "placeholder"), False)


                #os.rmdir(subpath)

            