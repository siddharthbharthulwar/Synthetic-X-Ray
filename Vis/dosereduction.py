import numpy as np
import matplotlib.pyplot as plt


original = np.load('Vis/125truth.npy')
output_single = np.load('Vis/125single.npy')
output_double = np.load('Vis/125double.npy')

f = plt.figure()
f.add_subplot(3, 4, 1)
plt.imshow(original[0], cmap = 'gray', vmin = 0, vmax = 1)

f.add_subplot(3, 4, 2)
plt.imshow(original[48], cmap = 'gray', vmin = 0, vmax = 1)

f.add_subplot(3, 4, 3)
plt.imshow(original[92], cmap = 'gray', vmin = 0, vmax = 1)

f.add_subplot(3, 4, 4)
plt.imshow(original[127], cmap = 'gray', vmin = 0, vmax = 1)




f.add_subplot(3, 4, 5)
plt.imshow(output_single[0], cmap = 'gray', vmin = 0, vmax = 1)

f.add_subplot(3, 4, 6)
plt.imshow(output_single[48], cmap = 'gray', vmin = 0, vmax = 1)

f.add_subplot(3, 4, 7)
plt.imshow(output_single[92], cmap = 'gray', vmin = 0, vmax = 1)

f.add_subplot(3, 4, 8)
plt.imshow(output_single[127], cmap = 'gray', vmin = 0, vmax = 1)





f.add_subplot(3, 4, 9)
plt.imshow(output_double[0], cmap = 'gray', vmin = 0, vmax = 1)

f.add_subplot(3, 4, 10)
plt.imshow(output_double[48], cmap = 'gray', vmin = 0, vmax = 1)

f.add_subplot(3, 4, 11)
plt.imshow(output_double[92], cmap = 'gray', vmin = 0, vmax = 1)

f.add_subplot(3, 4, 12)
plt.imshow(output_double[127], cmap = 'gray', vmin = 0, vmax = 1)

plt.show()