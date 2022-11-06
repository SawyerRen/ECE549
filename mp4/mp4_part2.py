# Part 2: Fundamental Matrix Estimation, Camera Calibration, Triangulation
## Fundamental Matrix Estimation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

##
## load images and match files for the first example
##

I1 = Image.open('MP3_part2_data/library1.jpg')
I2 = Image.open('MP3_part2_data/library2.jpg')
matches = np.loadtxt('MP3_part2_data/library_matches.txt')

# this is a N x 4 file where the first two numbers of each row
# are coordinates of corners in the first image and the last two
# are coordinates of corresponding corners in the second image:
# matches(i,1:2) is a point in the first image
# matches(i,3:4) is a corresponding point in the second image

N = len(matches)

##
## display two images side-by-side with matches
## this code is to help you visualize the matches, you don't need
## to use it to produce the results for the assignment
##

I3 = np.zeros((I1.size[1], I1.size[0] * 2, 3))
I3[:, :I1.size[0], :] = I1
I3[:, I1.size[0]:, :] = I2
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.imshow(np.array(I3).astype(float))
ax.plot(matches[:, 0], matches[:, 1], '+r')
ax.plot(matches[:, 2] + I1.size[0], matches[:, 3], '+r')
ax.plot([matches[:, 0], matches[:, 2] + I1.size[0]], [matches[:, 1], matches[:, 3]], 'r')
plt.show()


def fit_fundamental(matches, normalize=False):
    if normalize:
        matches, T1, T2 = normalize_matches(matches)

    n = len(matches)
    rows = np.zeros((n, 9))
    for i in range(n):
        u1, v1 = matches[i, 0: 2]
        u2, v2 = matches[i, 2: 4]
        rows[i] = [u1 * u2, v1 * u2, u2, u1 * v2, v1 * v2, v2, u1, v1, 1]

    U, s, V = np.linalg.svd(rows)
    F = V[len(V) - 1].reshape(3, 3)

    # enforce rank 2
    U, s, V = np.linalg.svd(F)
    new_s = np.diag(s)
    new_s[-1] = 0
    new_F = np.dot(U, np.dot(new_s, V))
    new_F = np.dot(np.dot(T2.T, new_F), T1)
    return new_F


def normalize_matches(matches):
    mean = np.mean(matches, axis=0)
    matches_mean = matches - mean
    total1 = 0
    total2 = 0
    for i in range(len(matches)):
        total1 += matches_mean[i, 0] ** 2 + matches_mean[i, 1] ** 2
        total2 += matches_mean[i, 2] ** 2 + matches_mean[i, 3] ** 2

    std1 = np.sqrt(total1 / (2 * N))
    std2 = np.sqrt(total2 / (2 * N))

    for i in range(len(matches)):
        matches_mean[i, 0] = matches_mean[i, 0] / std1
        matches_mean[i, 1] = matches_mean[i, 1] / std1
        matches_mean[i, 2] = matches_mean[i, 2] / std2
        matches_mean[i, 3] = matches_mean[i, 3] / std2

    matches = matches_mean
    T1 = [[1 / std1, 0, -1 * (1 / std1) * mean[0]],
          [0, 1 / std1, -1 * (1 / std1) * mean[1]],
          [0, 0, 1]]
    T2 = [[1 / std2, 0, -1 * (1 / std2) * mean[2]],
          [0, 1 / std2, -1 * (1 / std2) * mean[3]],
          [0, 0, 1]]

    return matches, T1, T2


##
## display second image with epipolar lines reprojected
## from the first image
##
# first, fit fundamental matrix to the matches
F = fit_fundamental(matches)  # this is a function that you should write
M = np.c_[matches[:, 0:2], np.ones((N, 1))].transpose()
L1 = np.matmul(F, M).transpose()  # transform points from
# the first image to get epipolar lines in the second image

# find points on epipolar lines L closest to matches(:,3:4)
l = np.sqrt(L1[:, 0] ** 2 + L1[:, 1] ** 2)
L = np.divide(L1, np.kron(np.ones((3, 1)), l).transpose())  # rescale the line
pt_line_dist = np.multiply(L, np.c_[matches[:, 2:4], np.ones((N, 1))]).sum(axis=1)
closest_pt = matches[:, 2:4] - np.multiply(L[:, 0:2], np.kron(np.ones((2, 1)), pt_line_dist).transpose())

# find endpoints of segment on epipolar line (for display purposes)
pt1 = closest_pt - np.c_[L[:, 1], -L[:, 0]] * 10  # offset from the closest point is 10 pixels
pt2 = closest_pt + np.c_[L[:, 1], -L[:, 0]] * 10

# display points and segments of corresponding epipolar lines
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.imshow(np.array(I2).astype(float))
ax.plot(matches[:, 2], matches[:, 3], '+r')
ax.plot([matches[:, 2], closest_pt[:, 0]], [matches[:, 3], closest_pt[:, 1]], 'r')
ax.plot([pt1[:, 0], pt2[:, 0]], [pt1[:, 1], pt2[:, 1]], 'g')
plt.show()


## Camera Calibration

def evaluate_points(M, points_2d, points_3d):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param points_2d: 2D points N x 2
    :param points_3d: 3D points N x 3
    :return:
    """
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u - points_2d[:, 0], v - points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

## Camera Centers


## Triangulation
