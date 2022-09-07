# -*- coding = utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_list = ['00125v.jpg',
              '00149v.jpg',
              '00153v.jpg',
              '00351v.jpg',
              '00398v.jpg',
              '01112v.jpg']


# get mean and std of a channel
def get_mean_std(c):
    h, w = c.shape
    mean = np.sum(c) / (h * w)
    std = np.sum((c - mean) ** 2) ** 0.5 / (h * w)
    return mean, std


# get zero mean normalized cross correlation of two channels
def get_ncc(c1, c2):
    mean1, std1 = get_mean_std(c1)
    mean2, std2 = get_mean_std(c2)
    h, w = c1.shape
    return np.sum((c1 - mean1) * (c2 - mean2)) / (h * w * std1 * std2)


# get best offset with ssd
def get_best_offset_ssd(c1, c2):
    off_x, off_y = 0, 0
    max = -float("inf")
    r = 15
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            ch1 = c1
            ch2 = np.roll(c2, i, axis=0)
            ch2 = np.roll(ch2, j, axis=1)
            diff = - np.sum((ch1 - ch2) ** 2)
            if diff > max:
                max = diff
                off_x = i
                off_y = j
    return off_x, off_y


# get best offset with ncc
def get_best_offset_ncc(c1, c2):
    off_x, off_y = 0, 0
    max = -float("inf")
    r = 15
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            ch1 = c1
            ch2 = np.roll(c2, i, axis=0)
            ch2 = np.roll(ch2, j, axis=1)
            ncc = get_ncc(ch1, ch2)
            if ncc > max:
                max = ncc
                off_x = i
                off_y = j
    return off_x, off_y


def get_best_offset(ch1, ch2, method):
    if method == 'ssd':
        return get_best_offset_ssd(ch1, ch2)
    elif method == 'ncc':
        return get_best_offset_ncc(ch1, ch2)


def align(input_image, image_name, method_list):
    height, width = input_image.shape
    height = int(height / 3)
    B = input_image[0 * height:1 * height, :]
    G = input_image[1 * height:2 * height, :]
    R = input_image[2 * height:3 * height, :]

    for method in method_list:
        off_Gx, off_Gy = get_best_offset(R, G, method)
        off_Bx, off_By = get_best_offset(R, B, method)
        print(method + ":" + image_name + "offset of green channel: ", off_Gx, off_Gy)
        print(method + ":" + image_name + "offset of blue channel: ", off_Bx, off_By)
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

        new_image_title = method + '_' + image_name
        mpimg.imsave(new_image_title, new_image)


if __name__ == '__main__':
    method_list = ['ssd', 'ncc']
    for filename in image_list:
        image_name = filename
        img = mpimg.imread("data/" + filename)
        input_image = img / np.max(img)
        align(input_image, image_name, method_list)
