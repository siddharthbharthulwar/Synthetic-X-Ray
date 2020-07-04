import os
from viewdicom import viewVolume, returnNumSlices

root = r"D:\Documents\School\2020-21\CT\LIDC-IDRI"

for path, subdirs, files in os.walk(root):
    counter = 0
    for name in files:
        if (name[-3: ] == 'dcm'):
            counter +=1
    if (counter > 5):
        returnNumSlices(path)
        print(path)
    