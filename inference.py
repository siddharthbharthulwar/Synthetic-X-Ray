from tensorflow.keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

ARCH_PATH = r"Networks\256\AE\model_arch.json"
WEIGHTS_PATH = r"Networks\256\AE\model_weights.h5"
INPUT_PATH = r"Data\In_New\0\0007.png"
OUTPUT_TRUTH_PATH = ""

with open(ARCH_PATH, 'r') as f:

    model = model_from_json(f.read())

model.load_weights(WEIGHTS_PATH)

data = cv.imread(INPUT_PATH, cv.IMREAD_GRAYSCALE)
plt.imshow(data, cmap = 'gray')
plt.show()

data = np.reshape(data, (1, 256, 256, 1))
result = model.predict(data)
result = np.reshape(result, (128, 128, 128))

plt.imshow(result[64], cmap = 'gray')
plt.show()