import os


root = r"D:\Documents\School\2020-21\CT\LIDC-IDRI"
dataset_count = 0

num_slices = []
with open('dirs.txt', 'w') as fl:
    for path, subdirs, files in os.walk(root):
        counter = 0
        for name in files:
            if (name[-3: ] == 'dcm'):
                counter +=1
        if (counter > 50):
            fl.write(path + "\n")
            print(path)