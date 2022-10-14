# -*- coding = utf-8 -*-

import cv2
import random
from PIL import Image
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np


def load_and_grayscale(root, path):
    image = cv2.imread(root + path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def load_and_normalize(root, path):
    image = cv2.imread(root + path)
    image = cv2.normalize(image.astype('float'), None,
                          0.0, 1.0, cv2.NORM_MINMAX)
    return image


def ransac_fitting(match_coords, threshold):
    max_iteration = 1000
    num_best_inliers = 0
    avg_residual = 0

    for i in range(max_iteration):
        rand_idx = random.sample(range(match_coords.shape[0]), k=4)
        pairs = match_coords[rand_idx]

        h = fit_homography(pairs)

        if np.linalg.matrix_rank(h) < 3:
            continue

        errors = get_errors(match_coords, h)
        idx = np.where(errors < threshold)[0]
        inliers = match_coords[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_h = h.copy()

            avg_residual = errors[idx].sum() / num_best_inliers

    print("Number of inliers: {}, Average residual: {}"
          .format(num_best_inliers, avg_residual))

    return best_inliers, best_h, avg_residual


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


def get_errors(pairs, H):
    n = len(pairs)

    p1 = np.concatenate((pairs[:, 0:2], np.ones((n, 1))), axis=1)
    p2 = pairs[:, 2:4]

    estimate_p2 = np.zeros((n, 2))
    for i in range(n):
        temp = np.matmul(H, p1[i])
        estimate_p2[i] = (temp / temp[2])[0:2]

    errors = np.linalg.norm(p2 - estimate_p2, axis=1) ** 2

    return errors


def plot_inlier_matches(inliers, root, path1, path2):
    image1 = Image.open(root + path1).convert('L')
    image2 = Image.open(root + path2).convert('L')
    image3 = np.zeros((image1.size[1], image1.size[0] * 2))
    image3[:, :image1.size[0]] = image1
    image3[:, image1.size[0]:] = image2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(image3).astype(float), cmap='gray')
    ax.plot(inliers[:, 0], inliers[:, 1], '+r')
    ax.plot(inliers[:, 2] + image1.size[0], inliers[:, 3], '+r')
    ax.plot([inliers[:, 0], inliers[:, 2] + image1.size[0]], [inliers[:, 1], inliers[:, 3]],
            'r', linewidth=0.4)
    plt.axis('off')
    plt.savefig(root + 'inlier_matches.jpg', format='jpg')
    plt.show()


def sift_descriptors(img):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(img, None)
    return keypoints, descriptor


def plot_save_sift(keypoints, img, root, path):
    out = img.copy()
    out = cv2.drawKeypoints(img, keypoints, out,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(out)
    plt.axis('off')
    plt.show()


def get_matched_pixels(threshold, kp1, kp2, desp1, desp2):
    pair_dist = distance.cdist(desp1, desp2, 'sqeuclidean')
    index1 = np.where(pair_dist < threshold)[0]
    index2 = np.where(pair_dist < threshold)[1]
    coord1 = np.array([kp1[idx].pt for idx in index1])
    coord2 = np.array([kp2[idx].pt for idx in index2])
    match_coords = np.concatenate((coord1, coord2), axis=1)

    return match_coords


def warp_left(img, H):
    h, w, z = img.shape
    corners = [[0, 0], [w, 0], [w, h], [0, h]]

    corners_new = []
    for corner in corners:
        corner = np.append(np.array(corner), 1)
        corners_new.append(np.matmul(H, corner))
    corners_new = np.array(corners_new).T

    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)

    height_new = int(round(abs(y_min) + h))
    width_new = int(round(abs(x_min) + w))
    size = (width_new, height_new)

    warped_img = cv2.warpPerspective(src=img, M=H, dsize=size)

    return warped_img, (x_min, y_min)


def move_right(img, translation):
    x_min = translation[0]
    y_min = translation[1]
    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    h, w, z = img.shape

    height_new = int(round(abs(y_min) + h))
    width_new = int(round(abs(x_min) + w))
    size = (width_new, height_new)

    moved_img = cv2.warpPerspective(src=img, M=translation_mat, dsize=size)

    return moved_img


def stitch_img(img_left, img_right, H):
    warped_l, translation = warp_left(img_left, H)
    moved_r = move_right(img_right, translation)

    black = np.zeros(3)

    for i in range(moved_r.shape[0]):
        for j in range(moved_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = moved_r[i, j, :]

            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass

    return warped_l[:moved_r.shape[0], :moved_r.shape[1], :]


def plot_save_stitch(stitched, root, path):
    cv2.imshow('stitched_image', stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    stitched = cv2.convertScaleAbs(stitched, alpha=255)
    cv2.imwrite(root + path, stitched)


def basic():
    threshold_match = 7000
    threshold_ransac = 0.5

    root = '/Users/sawyer_ren/Documents/uiuc2022fall/ece549/ECE549/mp3/part1/'
    path1 = 'data/left.jpeg'
    path2 = 'data/right.jpeg'
    im_left = load_and_grayscale(root, path1)
    im_right = load_and_grayscale(root, path2)

    kp_l, desp_l = sift_descriptors(im_left)
    kp_r, desp_r = sift_descriptors(im_right)
    plot_save_sift(kp_l, im_left, root, 'outputs/sift_left.jpg')
    plot_save_sift(kp_r, im_right, root, 'outputs/sift_right.jpg')

    match_coords = get_matched_pixels(threshold_match, kp_l, kp_r, desp_l, desp_r)

    inliers, H, avg_residual = ransac_fitting(match_coords, threshold_ransac)
    plot_inlier_matches(inliers.astype(int), root, path1, path2)

    im_l_color = load_and_normalize(root, path1)
    im_r_color = load_and_normalize(root, path2)
    stitched = stitch_img(im_l_color, im_r_color, H)
    plot_save_stitch(stitched, root, 'stitched.jpg')


def extra():
    threshold_match = 7000
    threshold_ransac = 0.5

    root = '/Users/sawyer_ren/Documents/uiuc2022fall/ece549/ECE549/mp3/part1/'
    images = ['hill', 'ledge', 'pier']
    im = images[2]

    path1 = 'part1_extra_credit/' + im + '/1.JPG'
    path2 = 'part1_extra_credit/' + im + '/2.JPG'
    im_1 = load_and_grayscale(root, path1)
    im_2 = load_and_grayscale(root, path2)

    kp_1, desp_1 = sift_descriptors(im_1)
    kp_2, desp_2 = sift_descriptors(im_2)
    plot_save_sift(kp_1, im_1, root, im + '_sift_1.jpg')
    plot_save_sift(kp_2, im_2, root, im + '_sift_2.jpg')

    match_coords = get_matched_pixels(threshold_match, kp_1, kp_2,
                                      desp_1, desp_2)

    inliers, h, avg_residual = ransac_fitting(match_coords, threshold_ransac)
    plot_inlier_matches(inliers.astype(int), root, path1, path2)

    im_1_color = load_and_normalize(root, path1)
    im_2_color = load_and_normalize(root, path2)
    stitched = stitch_img(im_1_color, im_2_color, h)
    plot_save_stitch(stitched, root, im + '_stitched_12.jpg')

    path12 = im + '_stitched_12.jpg'
    im_12 = load_and_grayscale(root, path12)

    path3 = 'part1_extra_credit/' + im + '/3.JPG'
    im_3 = load_and_grayscale(root, path3)

    kp_12, desp_12 = sift_descriptors(im_12)
    kp_3, desp_3 = sift_descriptors(im_3)
    plot_save_sift(kp_12, im_12, root, im + '_sift_12.jpg')
    plot_save_sift(kp_3, im_3, root, im + '_sift_3.jpg')

    match_coords = get_matched_pixels(threshold_match, kp_12, kp_3,
                                      desp_12, desp_3)

    inliers, h, avg_residual = ransac_fitting(match_coords, threshold_ransac)
    im_12_color = load_and_normalize(root, path12)
    im_3_color = load_and_normalize(root, path3)
    stitched = stitch_img(im_12_color, im_3_color, h)
    plot_save_stitch(stitched, root, im + '_stitched_123.jpg')


if __name__ == '__main__':
    # basic()
    extra()
    print("Done! :)")
