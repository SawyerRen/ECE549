# -*- coding = utf-8 -*-
# imports
import os
import sys
import glob
import re
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


#####################################
### Provided functions start here ###
#####################################

# Image loading and saving

def LoadFaceImages(pathname, subject_name, num_images):
    """
    Load the set of face images.
    The routine returns
        ambimage: image illuminated under the ambient lighting
        imarray: a 3-D array of images, h x w x Nimages
        lightdirs: Nimages x 3 array of light source directions
    """

    def load_image(fname):
        return np.asarray(Image.open(fname))

    def fname_to_ang(fname):
        yale_name = os.path.basename(fname)
        return int(yale_name[12:16]), int(yale_name[17:20])

    def sph2cart(az, el, r):
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    ambimage = load_image(
        os.path.join(pathname, subject_name + '_P00_Ambient.pgm'))
    im_list = glob.glob(os.path.join(pathname, subject_name + '_P00A*.pgm'))
    if num_images <= len(im_list):
        im_sub_list = np.random.choice(im_list, num_images, replace=False)
    else:
        print(
            'Total available images is less than specified.\nProceeding with %d images.\n'
            % len(im_list))
        im_sub_list = im_list
    im_sub_list.sort()
    imarray = np.stack([load_image(fname) for fname in im_sub_list], axis=-1)
    Ang = np.array([fname_to_ang(fname) for fname in im_sub_list])

    x, y, z = sph2cart(Ang[:, 0] / 180.0 * np.pi, Ang[:, 1] / 180.0 * np.pi, 1)
    lightdirs = np.stack([y, z, x], axis=-1)
    return ambimage, imarray, lightdirs


def save_outputs(subject_name, albedo_image, surface_normals):
    im = Image.fromarray((albedo_image * 255).astype(np.uint8))
    im.save("%s_albedo.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:, :, 0] * 128 + 128).astype(np.uint8))
    im.save("%s_normals_x.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:, :, 1] * 128 + 128).astype(np.uint8))
    im.save("%s_normals_y.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:, :, 2] * 128 + 128).astype(np.uint8))
    im.save("%s_normals_z.jpg" % subject_name)


# Plot the height map

def set_aspect_equal_3d(ax):
    """https://stackoverflow.com/questions/13685386"""
    """Fix equal aspect bug for 3D plots."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)
    plot_radius = max([
        abs(lim - mean_)
        for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
        for lim in lims
    ])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def display_output(albedo_image, height_map):
    fig = plt.figure()
    plt.imshow(albedo_image, cmap='gray')
    plt.axis('off')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.view_init(20, 20)
    X = np.arange(albedo_image.shape[0])
    Y = np.arange(albedo_image.shape[1])
    X, Y = np.meshgrid(Y, X)
    H = np.flipud(np.fliplr(height_map))
    A = np.flipud(np.fliplr(albedo_image))
    A = np.stack([A, A, A], axis=-1)
    ax.xaxis.set_ticks([])
    ax.xaxis.set_label_text('Z')
    ax.yaxis.set_ticks([])
    ax.yaxis.set_label_text('X')
    ax.zaxis.set_ticks([])
    ax.yaxis.set_label_text('Y')
    surf = ax.plot_surface(
        H, X, Y, cmap='gray', facecolors=A, linewidth=0, antialiased=False)
    set_aspect_equal_3d(ax)
    plt.show()


# Plot the surface normals

def plot_surface_normals(surface_normals):
    """
    surface_normals: h x w x 3 matrix.
    """
    fig = plt.figure()
    ax = plt.subplot(1, 3, 1)
    ax.axis('off')
    ax.set_title('X')
    im = ax.imshow(surface_normals[:, :, 0])
    ax = plt.subplot(1, 3, 2)
    ax.axis('off')
    ax.set_title('Y')
    im = ax.imshow(surface_normals[:, :, 1])
    ax = plt.subplot(1, 3, 3)
    ax.axis('off')
    ax.set_title('Z')
    im = ax.imshow(surface_normals[:, :, 2])
    plt.show()


#######################################
### Your implementation starts here ###
#######################################

def preprocess(ambimage, imarray):
    """
    preprocess the data:
        1. subtract ambient_image from each image in imarray.
        2. make sure no pixel is less than zero.
        3. rescale values in imarray to be between 0 and 1.
    Inputs:
        ambimage: h x w
        imarray: h x w x Nimages
    Outputs:
        processed_imarray: h x w x Nimages
    """
    processed_imarray = imarray - ambient_image[:, :, np.newaxis]
    processed_imarray[processed_imarray < 0] = 0
    processed_imarray = processed_imarray / 255
    return processed_imarray


def photometric_stereo(imarray, light_dirs):
    """
    Inputs:
        imarray:  h x w x Nimages
        light_dirs: Nimages x 3
    Outputs:
        albedo_image: h x w
        surface_norms: h x w x 3
    """
    h = imarray.shape[0]
    w = imarray.shape[1]
    n_images = imarray.shape[2]
    n_pix = h * w

    imarray = imarray.reshape(n_pix, n_images).transpose()

    results = np.linalg.lstsq(light_dirs, imarray)
    g = results[0]

    albedo_image = np.linalg.norm(g, axis=0)
    surface_normals = g / albedo_image

    surface_normals = surface_normals.transpose().reshape(h, w, 3)
    albedo_image = albedo_image.reshape(h, w)

    return albedo_image, surface_normals


def get_surface(surface_normals, integration_method):
    """
    Inputs:
        surface_normals:h x w x 3
        integration_method: string in ['average', 'column', 'row', 'random']
    Outputs:
        height_map: h x w
    """
    start = time.time()
    if integration_method == 'row':
        height_map = row(surface_normals)
    elif integration_method == 'column':
        height_map = column(surface_normals)
    elif integration_method == 'average':
        height_map = average(surface_normals)
    elif integration_method == 'random':
        height_map = random(surface_normals)

    end = time.time()
    print('Method used: ' + integration_method + '; Time elapsed: {} s.'.format(end - start))

    return height_map


def row(surface_normals):
    fx = surface_normals[:, :, 0] / surface_normals[:, :, 2]
    fy = surface_normals[:, :, 1] / surface_normals[:, :, 2]

    row_sum_x = np.cumsum(fx, axis=1)
    col_sum_y = np.cumsum(fy, axis=0)
    return row_sum_x[0] + col_sum_y


def column(surface_normals):
    fx = surface_normals[:, :, 0] / surface_normals[:, :, 2]
    fy = surface_normals[:, :, 1] / surface_normals[:, :, 2]

    row_sum_x = np.cumsum(fx, axis=1)
    col_sum_y = np.cumsum(fy, axis=0)
    return col_sum_y[:, 0][:, np.newaxis] + row_sum_x


def average(surface_normals):
    return (column(surface_normals) + row(surface_normals)) / 2


def random(surface_normals):
    fx = surface_normals[:, :, 0] / surface_normals[:, :, 2]
    fy = surface_normals[:, :, 1] / surface_normals[:, :, 2]

    row_sum_x = np.cumsum(fx, axis=1)
    col_sum_y = np.cumsum(fy, axis=0)
    h = surface_normals.shape[0]
    w = surface_normals.shape[1]
    height_map = np.zeros((h, w))

    n_paths = 25

    for y in range(h):
        for x in range(w):
            if x != 0 or y != 0:
                for path in range(n_paths):
                    zeros = [0] * x
                    ones = [1] * y
                    coins = np.array(zeros + ones)
                    np.random.shuffle(coins)
                    current_x = 0
                    current_y = 0
                    step = 0
                    cumsum = 0

                    while current_x < x or current_y < y:
                        if coins[step] == 0:
                            cumsum += fx[current_y, current_x]
                            current_x += 1
                        else:
                            cumsum += fy[current_y, current_x]
                            current_y += 1

                        step += 1

                    height_map[y, x] += cumsum
                height_map[y, x] = height_map[y, x] / n_paths
    return height_map


def LoadFaceImages_improved(pathname, subject_name, threshold):
    def load_image(fname):
        return np.asarray(Image.open(fname))

    def fname_to_ang(fname):
        yale_name = os.path.basename(fname)
        return int(yale_name[12:16]), int(yale_name[17:20])

    def sph2cart(az, el, r):
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    ambimage = load_image(
        os.path.join(pathname, subject_name + '_P00_Ambient.pgm'))
    im_list = glob.glob(os.path.join(pathname, subject_name + '_P00A*.pgm'))

    sub_list = []
    for file in im_list:
        im_arr = load_image(file)
        num_shadow = len(np.where(im_arr < 50)[0])
        ratio = num_shadow / im_arr.size
        if ratio < threshold:
            sub_list.append(file)

    sub_list.sort()
    imarray = np.stack([load_image(fname) for fname in sub_list], axis=-1)
    Ang = np.array([fname_to_ang(fname) for fname in sub_list])

    x, y, z = sph2cart(Ang[:, 0] / 180.0 * np.pi, Ang[:, 1] / 180.0 * np.pi, 1)
    lightdirs = np.stack([y, z, x], axis=-1)
    return ambimage, imarray, lightdirs


# Main function
if __name__ == '__main__':
    root_path = 'croppedyale/'
    subject_name = 'yaleB07'
    integration_method = 'random'
    save_flag = True

    full_path = '%s%s' % (root_path, subject_name)
    ambient_image, imarray, light_dirs = LoadFaceImages(full_path, subject_name,
                                                        64)

    processed_imarray = preprocess(ambient_image, imarray)

    albedo_image, surface_normals = photometric_stereo(processed_imarray,
                                                       light_dirs)
    # ambient_image, imarray, light_dirs = LoadFaceImages_improved(full_path, subject_name, 0.6)

    height_map = get_surface(surface_normals, integration_method)

    if save_flag:
        save_outputs(subject_name, albedo_image, surface_normals)

    plot_surface_normals(surface_normals)

    display_output(albedo_image, height_map)
