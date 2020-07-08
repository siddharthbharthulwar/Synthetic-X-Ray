import os
from isFlipped import isFlipped, flip180, isFlipped2


root = 'CXR'

for path, subdirs, files in os.walk(root):
    for name in subdirs:
        subpath = os.path.join(root, name)
        for spath, ssubdirs, sfiles in os.walk(subpath):

            flipped = 0
            for name in sfiles:

                if (name == '0.png'):

                    flipped = isFlipped2(os.path.join(subpath, name))

            if (flipped == 1):

                for name in sfiles:


                    flip180(os.path.join(subpath, name), False)

            elif (flipped == 3):

                os.rmdir(subpath)

            