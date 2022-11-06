import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random


def sift_descriptors(img):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(img, None)
    return keypoints, descriptor


def get_matched_pixels(threshold, kp1, kp2, desp1, desp2):
    dist = distance.cdist(desp1, desp2, 'sqeuclidean')
    index1 = np.where(dist < threshold)[0]
    index2 = np.where(dist < threshold)[1]
    coord1 = np.array([kp1[idx].pt for idx in index1])
    coord2 = np.array([kp2[idx].pt for idx in index2])
    match_coords = np.concatenate((coord1, coord2), axis=1)

    return match_coords


def fit_homography(pairs):
    rows = []

    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)

        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1] * p1[0], -p2[1] * p1[1], -p2[1] * p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1], -p2[0] * p1[2]]
        rows.append(row1)
        rows.append(row2)

    a = np.array(rows)

    U, s, V = np.linalg.svd(a)
    H = V[len(V) - 1].reshape(3, 3)
    H = H / H[2, 2]

    return H


def get_errors(matches, F):
    ones = np.ones((matches.shape[0], 1))
    p1 = np.concatenate((matches[:, 0:2], ones), axis=1)
    p2 = np.concatenate((matches[:, 2:4], ones), axis=1)

    F_p1 = np.dot(F, p1.T).T
    F_p2 = np.dot(F.T, p2.T).T

    p1_line2 = np.sum(p1 * F_p2, axis=1)[:, np.newaxis]
    p2_line1 = np.sum(p2 * F_p1, axis=1)[:, np.newaxis]
    d1 = np.absolute(p1_line2) / np.linalg.norm(F_p2, axis=1)[:, np.newaxis]
    d2 = np.absolute(p2_line1) / np.linalg.norm(F_p1, axis=1)[:, np.newaxis]

    return (d1 + d2) / 2


def normalize_matches(matches):
    mean = np.mean(matches, axis=0)
    matches_mean = matches - mean
    total1 = 0
    total2 = 0
    for i in range(len(matches)):
        total1 += matches_mean[i, 0] ** 2 + matches_mean[i, 1] ** 2
        total2 += matches_mean[i, 2] ** 2 + matches_mean[i, 3] ** 2
    N = len(matches)
    std1 = np.sqrt(total1 / (2 * N))
    std2 = np.sqrt(total2 / (2 * N))

    for i in range(len(matches)):
        matches_mean[i, 0] = matches_mean[i, 0] / std1
        matches_mean[i, 1] = matches_mean[i, 1] / std1
        matches_mean[i, 2] = matches_mean[i, 2] / std2
        matches_mean[i, 3] = matches_mean[i, 3] / std2

    matches = matches_mean
    T1 = np.array([[1 / std1, 0, -1 * (1 / std1) * mean[0]],
                   [0, 1 / std1, -1 * (1 / std1) * mean[1]],
                   [0, 0, 1]])
    T2 = np.array([[1 / std2, 0, -1 * (1 / std2) * mean[2]],
                   [0, 1 / std2, -1 * (1 / std2) * mean[3]],
                   [0, 0, 1]])

    return matches, T1, T2


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
    if normalize:
        new_F = np.dot(np.dot(T2.T, new_F), T1)
    return new_F


def ransac_fitting(match_coords, threshold):
    max_iteration = 1000
    num_best_inliers = 0
    avg_residual = 0

    for i in range(max_iteration):
        F = fit_fundamental(match_coords, normalize=True)

        errors = get_errors(match_coords, F)
        idx = np.where(errors < threshold)[0]
        inliers = match_coords[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_F = F.copy()

            avg_residual = errors[idx].sum() / num_best_inliers

    return best_inliers, best_F, avg_residual


def plot_inlier_matches(inliers, root, img_name):
    I1 = Image.open(root + 'MP4_part2_data/' + img_name + '1.jpg').convert('L')
    I2 = Image.open(root + 'MP4_part2_data/' + img_name + '2.jpg').convert('L')
    I3 = np.zeros((I1.size[1], I1.size[0] * 2))
    I3[:, :I1.size[0]] = I1
    I3[:, I1.size[0]:] = I2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I3).astype(float), cmap='gray')
    ax.plot(inliers[:, 0], inliers[:, 1], '+r')
    ax.plot(inliers[:, 2] + I1.size[0], inliers[:, 3], '+r')
    ax.plot([inliers[:, 0], inliers[:, 2] + I1.size[0]], [inliers[:, 1], inliers[:, 3]],
            'r', linewidth=0.4)
    plt.axis('off')
    plt.savefig(root + 'outputs/' + img_name + '_inlier_matches.jpg', format='jpg')
    plt.show()


if __name__ == '__main__':
    root = './'
    im_name = 'gaudi'
    I1 = Image.open('./MP4_part2_data/' + im_name + '1.jpg').convert('L')
    I2 = Image.open('./MP4_part2_data/' + im_name + '2.jpg').convert('L')

    im1 = np.array(I1)
    im2 = np.array(I2)
    t_match = 20000
    t_ransac = 0.4

    kp1, desp1 = sift_descriptors(im1)
    kp2, desp2 = sift_descriptors(im2)

    match_coords = get_matched_pixels(t_match, kp1, kp2,
                                      desp1, desp2)

    inliers, F, avg_residual = ransac_fitting(match_coords, t_ransac)
    print("Number of inliers: {}, Average residual: {}".format(len(inliers), avg_residual))

    plot_inlier_matches(inliers, root, im_name)
