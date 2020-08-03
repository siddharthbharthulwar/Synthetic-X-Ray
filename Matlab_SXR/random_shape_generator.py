'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import resize
'''
from itertools import product
import numpy as np
import argparse
import os
import time
import warnings

def calculate_radius(dimensions, center):
    smallest = np.max(dimensions)

    # Calculate the smallest difference between boundary and center
    for i in range(len(dimensions)):
        if dimensions[i]-center[i]<smallest:
            smallest = dimensions[i]-center[i]
        if center[i]<smallest:
            smallest = center[i]

    return smallest


def within_distance(center, radius, dim):
    actual_dis = np.sqrt(np.square(dim[0]-center[0])+np.square(dim[1]-center[1])+np.square(dim[2]-center[2]))
    return True if actual_dis <= radius else False


def on_border(center, radius, dim):
    actual_dis = np.sqrt(np.square(dim[0]-center[0])+np.square(dim[1]-center[1])+np.square(dim[2]-center[2]))
    return True if actual_dis <= radius and actual_dis > radius - 1 else False


def create_sphere(matrix, center, radius, value, border_points):
    dimension = len(matrix.shape)
    shape = matrix.shape
    for x, y, z in product(range(shape[0]), range(shape[1]), range(shape[2])):
        dim = [x,y,z]
        if within_distance(center, radius, dim) and matrix[x,y,z] != value:
            matrix[x,y,z] = value
            if on_border(center, radius, dim):
                border_points.append(dim)
    print(np.sum(matrix))
    return border_points


''' OLD METHOD
def create_sphere(matrix, center, radius, value, pts_on_border):
    dimension = len(matrix.shape)
    shape = matrix.shape
    inside_boder = False
    cross_border = False
    count = 0

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                dim = [x,y,z]
                if within_distance(center, radius, dim):
                    if not inside_border:
                        cross_border = True
                    if cross_border and matrix[x,y,z] != value and on_border(center, radius, dim):
                        pts_on_border.append((x,y,z))
                        # cross_border = False
                    matrix[x,y,z] = value
                    inside_border = True
                else:
                    inside_border = False

                    if cross_border:
                        cross_border = False
                        continue
                if matrix[x,y,z] == value:
                    count = count + 1
    print(count)
    return pts_on_border
'''


def display_nodule(matrix):
    # Plot three perspectives
    fig_0 = plt.figure(0)
    plt.imshow(np.sum(matrix, 0), cmap='gray')
    fig_1 = plt.figure(1)
    plt.imshow(np.sum(matrix, 1), cmap='gray')
    fig_2 = plt.figure(2)
    plt.imshow(np.sum(matrix, 2), cmap='gray')
    plt.show()

    z,x,y = matrix.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c= 'red')
    plt.show()


def find_new_center_radius(cen_rad_tuple, border_pts, dimensions, size):
    """Finds a point on the surface of the sphere specified by the tuple.

    """
    new_cen = border_pts[np.random.randint(len(border_pts))]
    new_rad = np.random.uniform(0.6, 0.85)*calculate_radius(dimensions, new_cen)

    return new_cen, new_rad


def shape_grower(size, voxel_dimension):
    #larger_dimensions = [int(1.2*size/(voxel_dimensions[0]/10)), int(1.2*size/(voxel_dimensions[1]/10)), int(1.05*size/(voxel_dimensions[2]/10))]
    larger_dimensions = 3*[int(size/voxel_dimension)+3]

    dimensions = 3*[int(size/voxel_dimension)]
    print(dimensions)
    base = np.zeros(larger_dimensions)
    print(base.shape)

    # Calculate the center
    center = ()
    for i in range(len(larger_dimensions)):
        center += (int(larger_dimensions[i]/2),)

    # Choose radius for initial center
    radius = np.random.uniform(0.5, 0.75)*calculate_radius(larger_dimensions, center)

    # Create list of tuples of center and radius
    cen_rads = []
    cen_rads.append((center, radius))
    border_pts = []
    internal_points = []

    i = 0
    while not reach_desired_size(base, dimensions):
        # Poll a tuple from cen_rads list
        cen_rad_tuple = cen_rads[i]

        # Create sphere in the matrix
        print("Center is {} and radius is {}".format(cen_rad_tuple[0], cen_rad_tuple[1]))
        # border_pts = ConvexHull(create_sphere(base, cen_rad_tuple[0], cen_rad_tuple[1], 1, all_points)).vertices #list(set(create_sphere(base, cen_rad_tuple[0], cen_rad_tuple[1], 1, border_pts)))

        border_pts = create_sphere(base, cen_rad_tuple[0], cen_rad_tuple[1], 1, border_pts)

        print(len(border_pts))

        i += 1
        if reach_desired_size(base, dimensions):
            break

        # Create a new center and new radius
        new_cen, new_rad = find_new_center_radius(cen_rad_tuple, border_pts, larger_dimensions, size)
        cen_rads.append((new_cen, new_rad))

    return base


def reach_desired_size(base, dimensions):
    print(dimensions)
    print(np.amax(np.sum(base, axis=0)), np.amax(np.sum(base, axis=1)), np.amax(np.sum(base, axis=2)))
    reach_x = np.amax(np.sum(base, axis=0)) >= dimensions[0]
    reach_y = np.amax(np.sum(base, axis=1)) >= dimensions[1]
    reach_z = np.amax(np.sum(base, axis=2)) >= dimensions[2]
    return reach_x or reach_y or reach_z


def main():
    np.random.seed(int(time.time()))
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action="store", dest="vox_dim", type=float, help="Dimensions of a voxel: d")
    parser.add_argument('-s', action="store", dest="size_cm", type=float, help="Size of nodule (cm): s")
    args = parser.parse_args()

    voxel_dimension = args.vox_dim/10 # convert from mm to cm

    print(voxel_dimension)

    size = args.size_cm

    print(size)

    result_path = './numpy_nodules'
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    file_prefix = 'nodule_'
    i = len(os.listdir(result_path))

    matrix = shape_grower(size, voxel_dimension)
    np.save(os.path.join(result_path, file_prefix+str(i+1)+'.npy'), matrix)

    #matrix = np.asarray(matrix, dtype='uint8')
    #dimensions = (int(size/(voxel_dimensions[0]/10)), int(size/(voxel_dimensions[1]/10)), int(size/(voxel_dimensions[2]/10)))
    #matrix = resize(matrix, dimensions, order=1, preserve_range=True, mode='constant')
    #display_nodule(matrix)


if __name__ == "__main__":
    main()
