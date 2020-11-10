import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

truth = np.load(r'Vis\7org.npy')
single = np.load(r'Vis\7single.npy')
double = np.load(r'Vis\7double.npy')

def scale(x, out_range=(-1, 1)):
    domain = 0, 1
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


truth_hist = np.ravel(scale(truth, (-1000, 3000)))
single_hist = np.ravel(scale(single, (-1000, 3000)))
double_hist = np.ravel(scale(double, (-1000, 3000)))

plt.hist(truth_hist, bins = 2000, alpha = 0.3, label = "Truth")
plt.hist(single_hist, bins = 2000, alpha = 0.5, label = "Single CNN")
plt.hist(double_hist, bins = 2000, alpha = 0.3, label = "Dual CNN")
plt.xlabel("Hounsfield Unit (HU)")
plt.ylabel("Frequency")
plt.xlim(-1050, 1050)
plt.ylim(0, 15000)
plt.legend()
plt.show()



'''
truu = np.ravel(truth)
singg = np.ravel(single)
dubb = np.ravel(double)

plt.hist(truth_hist, bins = 2000, alpha = 0.3, label = "Truth")
plt.hist(single_hist, bins = 2000, alpha = 0.5, label = "Single CNN")
plt.hist(double_hist, bins = 2000, alpha = 0.3, label = "Dual CNN")

print(cv.compareHist())
'''

'''
print(truu.shape)
print(singg.shape)
print(dubb.shape)

plt.imshow(np.abs(truu - singg)[64], cmap = 'jet')
plt.colorbar(orientation="horizontal",fraction=0.07,anchor=(1.0,0.0))
plt.show()

plt.imshow(np.abs(truu - dubb)[64], cmap = 'jet')
plt.colorbar(orientation="horizontal",fraction=0.07,anchor=(1.0,0.0))
plt.show()

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print(100 - mean_absolute_error(truu, singg))
print(100 - mean_absolute_error(truu, dubb))
print(mean_squared_error(scale(truth, (-1000, 3000)), scale(single, (-1000, 3000))))
print(mean_squared_error(scale(truth, (-1000, 3000)), scale(double, (-1000, 3000))))
'''