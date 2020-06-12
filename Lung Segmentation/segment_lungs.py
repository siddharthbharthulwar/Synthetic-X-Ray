from PIL import Image
import numpy as np
import operator


def read_ct(filename):
    f = open(filename)
    ct = []
    for line in f:
        slice_list = line.strip().split(",")
        ct_slice = []
        for i in range(512):
            row = []
            for j in range(512):
                grayscale = float(slice_list[512 * i + j])
                row.append([grayscale] * 3)
            ct_slice.append(row)
        ct.append(ct_slice)
    return ct

def label_slice(slice, threshold, pts):

    def check_point(slice, q, visited, x, y):
        # print slice
        if (x, y) not in visited and x <= slice.shape[0] and x >= 0 and y >= 0 and y <= slice.shape[1] and slice[x, y, 0] <= threshold:
            q.append((x, y))
            slice[x, y] = [255, 0, 0]

        visited[(x, y)] = 1

    real_pts = [(y, x) for x, y in pts]
    copy = slice.copy()
    if slice.ndim > 3:
        print "> 3 dims"
        copy = slice[:]
    visited = {}
    q = real_pts
    for x, y in q:
        copy[x, y] = [0, 0, 255]
        visited[(x, y)] = 1

    while len(q) > 0:
        # check_pts(slice, threshold, pts, pts_to_check, z, x, y)
        (x, y) = q.pop()


        check_point(copy, q, visited, x + 1, y)
        check_point(copy, q, visited, x, y + 1)
        check_point(copy, q, visited, x - 1, y)
        check_point(copy, q, visited, x, y - 1)

    return copy, visited

# def label_slice_3d(ct, threshold, q, visited):
#     def check_point(ct, q, visited, slice_visited, pt):
#         if pt not in visited and \
#                 0 <= pt[0] <= ct.shape[0] - 1 and \
#                 0 <= pt[1] <= ct.shape[1] - 1 and \
#                 0 <= pt[2] <= ct.shape[2] - 1 and \
#                 33 <= ct[pt + (0,)] <= threshold:
#             q.append(pt)
#             slice_visited[pt] = 1
#         visited[pt] = 1
#
#     slice_visited = {}
#     while len(q) > 0:
#         slice_pt = q.pop()
#
#         check_point(ct, q, visited, slice_visited, tuple(map(operator.add, slice_pt, (0, 0, 1))))
#         check_point(ct, q, visited, slice_visited, tuple(map(operator.add, slice_pt, (0, 1, 0))))
#         check_point(ct, q, visited, slice_visited, tuple(map(operator.add, slice_pt, (0, 0, -1))))
#         check_point(ct, q, visited, slice_visited, tuple(map(operator.add, slice_pt, (0, -1, 0))))
#     return slice_visited
#
# def segment_lungs(ct, threshold, pts_both_lungs):
#     """
#
#     :param ct: np array of dim 4 (last one is grayscale)
#     :param threshold:
#     :param pts: list of 2 lists containing starting points from each lung
#     :return: list of pts in lungs
#     """
#     def segment_lung(ct, threshold, pts, lr):
#         q = pts
#         visited = {}
#         lung_pts = {}
#         slice_visited = label_slice_3d(ct, threshold, q, visited)
#         q = next_slice_pts(slice_visited.keys(), lr)
#
#         print "{} points in first slice_visited".format(len(slice_visited))
#         print "{} points in lung_pts".format(len(lung_pts))
#         print "{} points in first q".format(len(q))
#
#         i = 0
#         while len(slice_visited) > 1:
#
#             slice_visited = label_slice_3d(ct, threshold, q, visited)
#
#             q = next_slice_pts(slice_visited.keys(), lr)
#             lung_pts.update(slice_visited)
#             print "{} points in q".format(len(q))
#             print "{} points in slice_visited".format(len(slice_visited))
#             print "{} points in lung_pts".format(len(lung_pts))
#             print "iteration {} done".format(i)
#             i += 1
#
#         return lung_pts
#
#     left_pts = segment_lung(ct, threshold, pts_both_lungs[1], "l")
#     print "==== left slice done ===="
#     right_pts = segment_lung(ct, threshold, pts_both_lungs[0], "r")
#     print "==== right slice done ===="
#     left_pts.update(right_pts)
#
#     return left_pts.keys()


# def next_slice_pts(pt_list, lr):
#     above_below = [(z - 1, x, y) for (z, x, y) in pt_list] + \
#                   [(z + 1, x, y) for (z, x, y) in pt_list]
#     min_x = 512
#     max_x = 0
#     for pt in above_below:
#         # print pt[1]
#         if pt[1] < min_x:
#             min_x = pt[1]
#         if pt[1] > max_x:
#             max_x = pt[1]
#     # print min_x
#     l = []
#     line = (max_x + min_x)/2
#     # print line, "===="
#     for pt in above_below:
#         # print pt[1], "----"
#         if lr == "l" and pt[1] < line:
#             l.append(pt)
#         elif lr == "r" and pt[1] > line:
#             l.append(pt)
#     return l



def segment_lungs_bfs(ct, l_threshold, u_threshold, pts):
    def check_point(pt):
        if pt not in visited and \
                0 <= pt[0] <= ct.shape[0] - 1 and \
                0 <= pt[1] <= ct.shape[1] - 1 and \
                0 <= pt[2] <= ct.shape[2] - 1 and \
                l_threshold <= ct[pt + (0,)] <= u_threshold:
            q.append(pt)
            selected[pt] = 1
            if len(selected) % 100000 == 0:
                print "{} points selected".format(len(selected))
        visited[pt] = 1

    visited = {}
    selected = {}
    q = pts
    while len(q) > 0:
        if len(q) % 100000 == 0:
            print "{}E4 points in q".format(len(q)/10000)
        slice_pt = q.pop()

        check_point(tuple(map(operator.add, slice_pt, (0, 0, 1))))
        check_point(tuple(map(operator.add, slice_pt, (0, 1, 0))))
        check_point(tuple(map(operator.add, slice_pt, (0, 0, -1))))
        check_point(tuple(map(operator.add, slice_pt, (0, -1, 0))))
        check_point(tuple(map(operator.add, slice_pt, (1, 0, 0))))
        check_point(tuple(map(operator.add, slice_pt, (-1, 0, 0))))
    return selected


if __name__ == "__main__":

    # SLICE ORIENTATION
    # +------> y
    # |
    # |
    # |
    # v
    # x
    print "reading ct..."
    ct = np.array(read_ct("../CTs/CT_3.txt"))
    print "done reading ct"
    max = np.amax(ct)

    print "max: ", max
    norm_ct = ct / max * 255
    l_threshold = 33
    u_threshold = 500 / max * 255
    norm_ct = norm_ct.astype(np.uint8)

    # z, x, y = 90, 255, 150
    left = [(90, 150, 255)]
    right = [(90, 350, 255)]
    # start_pts = [left, right]
    print "upper threshold: {}, left: {}, right: {}".format(u_threshold, norm_ct[left[0]], norm_ct[right[0]])
    # norm_ct[z], visited = label_slice(norm_ct[z], threshold, [(x, y)])

    # pts = segment_lungs(norm_ct, threshold, start_pts)
    lung_pts = segment_lungs_bfs(norm_ct, l_threshold, u_threshold, left + right)
    print "{} of {} ct points selected ({}%)".format(len(lung_pts), ct.size, len(lung_pts) / float(ct.size) * 100)

    print "writing to file"
    f = open("lung_pts.txt", "w")
    for pt in lung_pts:
        f.write("{},{},{}\n".format(pt[0], pt[1], pt[2]))
    f.close()
    print "done"
    # w, h = 512, 512
    # data = np.zeros((h, w, 3), dtype=np.uint8)
    # data[256, 256] = [255, 0, 0]
    # print data
    # img = Image.fromarray(data, 'RGB')

    # image = Image.fromarray(norm_ct[z])
    # image.show()
