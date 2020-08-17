import matplotlib.pyplot as plt
from PIL import Image
import os

'''

image = Image.open(r"Data\In\0\0001.png")
new_image = image.resize((256, 256))
new_image.show()

'''

for img in os.listdir(r"Data\In\0"):

    image = Image.open(os.path.join(r"Data\In\0", img))
    filename = img[0:4]
    new_image = image.resize((256, 256))
    path = os.path.join(r"Data\In_New\0", filename + ".png")
    new_image.save(path)
    print(filename)

for img in os.listdir(r"Data\In\1"):

    image = Image.open(os.path.join(r"Data\In\1", img))
    filename = img[0:4]
    new_image = image.resize((256, 256))
    path = os.path.join(r"Data\In_New\1", filename + ".png")
    new_image.save(path)
    print(filename)

for img in os.listdir(r"Data\In\2"):

    image = Image.open(os.path.join(r"Data\In\2", img))
    filename = img[0:4]
    new_image = image.resize((256, 256))
    path = os.path.join(r"Data\In_New\2", filename + ".png")
    new_image.save(path)
    print(filename)

for img in os.listdir(r"Data\In\3"):

    image = Image.open(os.path.join(r"Data\In\3", img))
    filename = img[0:4]
    new_image = image.resize((256, 256))
    path = os.path.join(r"Data\In_New\3", filename + ".png")
    new_image.save(path)
    print(filename)