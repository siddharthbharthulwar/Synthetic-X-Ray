import os
import numpy as np

CTDirlist = os.listdir(r"Data\Out")
XrayDirList = os.listdir(r"CXR")

CTDirlistNew = []
for i in CTDirlist:

    CTDirlistNew.append(i[0:4])

main_list = np.setdiff1d(CTDirlistNew, XrayDirList)

for item in main_list:

    path = os.path.join(r"Data\Out", item + ".npy")
    print(path)
