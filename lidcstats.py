import os
from viewdicom import viewVolume, returnNumSlices
import matplotlib.pyplot as plt
import statistics

'''

root = r"D:\Documents\School\2020-21\CT\LIDC-IDRI"

slices = []

for path, subdirs, files in os.walk(root):
    counter = 0
    for name in files:
        if (name[-3: ] == 'dcm'):
            counter +=1
    if (counter > 5):
        slices.append(returnNumSlices(path))
        print(path)

n, bins, patches = plt.hist(slices, 100, facecolor = 'blue', alpha = 0.8)
plt.show()


'''
slices = []

with open ('dirs.txt') as f:

    content = f.readlines()

content = [x.strip() for x in content]

for file in content:

    slices.append(len(os.listdir(file)))

L2 = [ x for x in slices if 50 <= x <= 350 ]
print(statistics.mean(L2))
'''
n, bins, patches = plt.hist(slices, 350, facecolor = 'blue', alpha = 0.7)
plt.xlabel('Number of Slices')
plt.ylabel('Frequency (Number of Scans)')
plt.title('Distribution of Slices in LIDC-IDRI')
plt.show()    
'''
