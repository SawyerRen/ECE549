# -*- coding = utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import time

image_list = [
    '00149v.jpg',
    '00153v.jpg',
    '00351v.jpg',
    '00398v.jpg',
    '01112v.jpg']
image_list2 = ['01047u',
               '01657u',
               '01861a']


def get_best_offset_fourier(c1, c2, channel_name):
    off_x, off_y = 0, 0
    max = -float("inf")

    ch1 = gaussian_filter(c1, sigma=1)
    ch2 = gaussian_filter(c2, sigma=1)
    ft1 = np.fft.fft2(ch1)
    ft1_shift = np.fft.fftshift(ft1)
    # ft1_res = np.log(np.abs(ft1_shift))
    # plt.imshow(ft1_res)
    # plt.show()
    ft2 = np.fft.fft2(ch2)
    ft2_shift = np.fft.fftshift(ft2)
    # ft2_res = np.log(np.abs(ft2_shift))
    # plt.imshow(ft2_res)
    # plt.show()
    ft2_conjugate = np.conjugate(ft2_shift)
    prod = ft1_shift * ft2_conjugate
    con = np.fft.ifft2(prod)
    con_res = np.log(np.abs(con))
    plt.imshow(con_res)
    if channel_name == 'G':
        plt.title('G to R alignment with preprocessing')
    else:
        plt.title('B to R alignment with preprocessing')
    plt.show()
    rows = con.shape[0]
    cols = con.shape[1]

    for row in range(rows):
        for col in range(cols):
            if con[row][col] > max:
                max = con[row][col]
                off_x = row
                off_y = col

    return off_x, off_y


def get_best_offset_fourier_without_filter(c1, c2, channel_name):
    ch1 = c1
    ch2 = c2
    ft1 = np.fft.fft2(ch1)
    ft1_shift = np.fft.fftshift(ft1)

    ft2 = np.fft.fft2(ch2)
    ft2_shift = np.fft.fftshift(ft2)

    ft2_conjugate = np.conjugate(ft2_shift)
    prod = ft1_shift * ft2_conjugate
    con = np.fft.ifft2(prod)
    con_res = np.log(np.abs(con))
    plt.imshow(con_res)
    if channel_name == 'G':
        plt.title('G to R alignment without preprocessing')
    else:
        plt.title('B to R alignment without preprocessing')
    plt.show()


def align(input_image, image_name, extension):
    start_time = time.time()
    height, width = input_image.shape
    height = int(height / 3)
    B = input_image[0 * height:1 * height, :]
    G = input_image[1 * height:2 * height, :]
    R = input_image[2 * height:3 * height, :]
    # get_best_offset_fourier_without_filter(R, G, 'G')
    # get_best_offset_fourier_without_filter(R, B, 'B')
    off_Gx, off_Gy = get_best_offset_fourier(R, G, 'G')
    off_Bx, off_By = get_best_offset_fourier(R, B, 'B')
    print(image_name + " offset of green channel: ", off_Gx, off_Gy)
    print(image_name + " offset of blue channel: ", off_Bx, off_By)
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
    plt.imshow(new_image)
    plt.title('result image')
    plt.show()
    if extension:
        new_image_title = 'F_' + image_name
    else:
        new_image_title = 'F_' + image_name + ".png"
    mpimg.imsave(new_image_title, new_image)
    end_time = time.time()
    print(image_name + " time = ", end_time - start_time)


if __name__ == '__main__':
    # for filename in image_list:
    #     image_name = filename
    #     img = mpimg.imread("data/" + filename)
    #     input_image = img / np.max(img)
    #     align(input_image, image_name, True)
    for filename in image_list2:
        image_name = filename
        filename = filename + ".tif"
        img = mpimg.imread("data_hires/" + filename)
        input_image = img / np.max(img)
        align(input_image, image_name, False)
