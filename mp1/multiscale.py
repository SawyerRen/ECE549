# -*- coding = utf-8 -*-
# -*- coding = utf-8 -*-
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

image_list = ['01047u',
              '01657u',
              '01861a']


# get mean and std of a channel
def get_mean_std(c):
    h, w = c.shape
    mean = np.sum(c) / (h * w)
    std = np.sum((c - mean) ** 2) ** 0.5 / (h * w)
    return mean, std


# get zero mean normalized cross correlation of two channels
def get_zncc(c1, c2):
    mean1, std1 = get_mean_std(c1)
    mean2, std2 = get_mean_std(c2)
    h, w = c1.shape
    return np.sum((c1 - mean1) * (c2 - mean2)) / (h * w * std1 * std2)


# get best offset with ssd
def get_best_offset_ssd(c1, c2, r=15, off_x=0, off_y=0):
    temp_x = off_x
    temp_y = off_y
    max_metric = -float("inf")
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            ch1 = c1
            ch2 = np.roll(c2, temp_x + i, axis=0)
            ch2 = np.roll(ch2, temp_y + j, axis=1)
            diff = - np.sum((ch1 - ch2) ** 2)
            if diff > max_metric:
                max_metric = diff
                off_x = temp_x + i
                off_y = temp_y + j
    return off_x, off_y


# get best offset with zncc
def get_best_offset_zncc(c1, c2, r=15, off_x=0, off_y=0):
    max_metric = -float("inf")
    temp_x = off_x
    temp_y = off_y
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            ch1 = c1
            ch2 = np.roll(c2, temp_x + i, axis=0)
            ch2 = np.roll(ch2, temp_y + j, axis=1)
            zncc = get_zncc(ch1, ch2)
            if zncc > max_metric:
                max_metric = zncc
                off_x = temp_x + i
                off_y = temp_y + j
    return off_x, off_y


def get_best_offset(ch1, ch2, method, r=15, off_x=0, off_y=0):
    if method == 'ssd':
        return get_best_offset_ssd(ch1, ch2, r, off_x, off_y)
    elif method == 'zncc':
        return get_best_offset_zncc(ch1, ch2, r, off_x, off_y)


def get_best_offset_multi_scale(ch1, ch2, method):
    d1 = cv.pyrDown(ch1)
    d2 = cv.pyrDown(d1)
    d3 = cv.pyrDown(d2)
    d4 = cv.pyrDown(d3)
    d5 = cv.pyrDown(ch2)
    d6 = cv.pyrDown(d5)
    d7 = cv.pyrDown(d6)
    d8 = cv.pyrDown(d7)
    r = 80
    pyramid_list1 = [d4, d3, d2, d1]
    pyramid_list2 = [d8, d7, d6, d5]
    off_x, off_y = 0, 0
    for i in range(len(pyramid_list1)):
        off_x, off_y = get_best_offset(pyramid_list1[i], pyramid_list2[i], method, r, off_x, off_y)
        off_x = off_x * 2
        off_y = off_y * 2
        r = int(r / 2)
    return off_x, off_y


def align(input_image, image_name, method_list):
    height, width = input_image.shape
    height = int(height / 3)
    R = input_image[2 * height:3 * height, :]
    G = input_image[1 * height:2 * height, :]
    B = input_image[0 * height:1 * height, :]

    for method in method_list:

        off_Gx, off_Gy = get_best_offset_multi_scale(R, G, method)
        off_Bx, off_By = get_best_offset_multi_scale(R, B, method)

        R = R
        G = np.roll(G, off_Gx, axis=0)
        G = np.roll(G, off_Gy, axis=1)
        B = np.roll(B, off_Bx, axis=0)
        B = np.roll(B, off_By, axis=1)

        new_height, new_width = R.shape
        new_image = np.zeros((new_height, new_width, 3))
        new_image[:, :, 0] = R
        new_image[:, :, 1] = G
        new_image[:, :, 2] = B

        new_image_title = method + '_' + image_name + "_multi.png"
        mpimg.imsave(new_image_title, new_image)


if __name__ == '__main__':
    method_list = ['ssd', 'zncc']
    for filename in image_list:
        image_name = filename
        filename = filename + ".tif"
        img = mpimg.imread("data_hires/" + filename)
        input_image = img / np.max(img)
        align(input_image, image_name, method_list)
