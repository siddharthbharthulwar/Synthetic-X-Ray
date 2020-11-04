from skimage.draw import circle
import numpy as np
import matplotlib.pyplot as plt

x = np.zeros((128, 128))
rr, cc = circle(65, 71, 42.8524180)
x[rr, cc] = 1


index = 64
slicc = x[index]

def get_first(slice, target):

    for i in range(0, len(slice)):

        if (slice[i] == target):

            return i

def get_last(slice, target):

    for i in range(len(slice) - 1, 0, -1):

        if (slice[i] == target):
            
            return i

print(get_first(slicc, 1))
print(get_last(slicc, 1))