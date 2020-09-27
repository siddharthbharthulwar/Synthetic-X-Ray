import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
path = r'Data\Out\0001.npy'
ims = []
array = np.load(path)

for i in range(0, 256):

    slice = array[i]
    im = plt.imshow(slice, cmap = 'gray', vmin = np.amin(array), vmax = np.amax(array))
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval =50, blit = True, repeat_delay = 1000)
ani.save('ct_animation.gif')
plt.show()