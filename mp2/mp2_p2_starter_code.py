# Libraries you will find useful
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from skimage import transform
from scipy.ndimage.filters import gaussian_laplace, convolve
from scipy.ndimage.filters import rank_filter, generic_filter
import time
import math
from itertools import chain
from matplotlib.patches import Circle


def get_gray_image(file_path):
    print(file_path)
    img = io.imread(file_path)
    plt.imshow(img)
    plt.show()
    gray_img = color.rgb2gray(img)
    gray_img /= 255
    return gray_img


def increase_filter_size(image, init_sigma, levels, k):
    sigma = np.zeros(levels)
    sigma[0] = init_sigma
    h = image.shape[0]
    w = image.shape[1]
    scale_space = np.empty((h, w, levels))
    start = time.time()
    for i in range(levels):
        filter_size = int(round(6 * sigma[i]))
        if filter_size % 2 == 0:
            filter_size += 1
        center = int((filter_size + 1) / 2 - 1)
        gauss_filter = np.zeros((filter_size, filter_size))
        gauss_filter[center][center] = 1
        LoG = sigma[i] * sigma[i] * gaussian_laplace(gauss_filter, sigma[i])
        scale_space[:, :, i] = convolve(image, LoG) ** 2

        if i + 1 < levels:
            sigma[i + 1] = sigma[i] * k
    end = time.time()
    print('increase_filter_size: Time : {} s.'.format(end - start))
    return scale_space, sigma


def down_sample(image, init_sigma, levels, k):
    sigma = np.zeros(levels)
    sigma[0] = init_sigma
    h = image.shape[0]
    w = image.shape[1]
    scale_space = np.empty((h, w, levels))
    start = time.time()
    filter_size = int(round(6 * sigma[0]))
    if filter_size % 2 == 0:
        filter_size += 1
    center = int((filter_size + 1) / 2 - 1)
    gauss_filter = np.zeros((filter_size, filter_size))
    gauss_filter[center][center] = 1
    LoG = gaussian_laplace(gauss_filter, sigma[0])
    for i in range(levels):
        scaled_im = transform.rescale(image, (1 / k) ** i, order=3)
        im_tmp = convolve(scaled_im, LoG) ** 2
        scale_space[:, :, i] = transform.resize(im_tmp, (h, w), order=3)
        if (i + 1 < levels):
            sigma[i + 1] = sigma[i] * k
    end = time.time()
    print('down_sample: Time : {} s.'.format(end - start))
    return scale_space, sigma


def get_scale_space(img, init_sigma, levels, k, method):
    if method == "increase":
        return increase_filter_size(img, init_sigma, levels, k)
    else:
        return down_sample(img, init_sigma, levels, k)


def non_maximum_suppression_rank(scale_space):
    start = time.time()
    h = scale_space.shape[0]
    w = scale_space.shape[1]
    levels = scale_space.shape[2]
    local_max = np.empty((h, w, levels))
    for i in range(levels):
        curr_response = scale_space[:, :, i]
        local_max[:, :, i] = rank_filter(curr_response, -1, (3, 3))
    end = time.time()
    print('rank filter: Time : {} s.'.format(end - start))
    return local_max


def non_maximum_suppression_generic(scale_space):
    start = time.time()
    h = scale_space.shape[0]
    w = scale_space.shape[1]
    levels = scale_space.shape[2]
    local_max = np.empty((h, w, levels))
    for i in range(levels):
        curr_response = scale_space[:, :, i]
        find_max = lambda arr: np.amax(arr)
        local_max[:, :, i] = generic_filter(curr_response, find_max, (3, 3))
    end = time.time()
    print('generic filter: Time : {} s.'.format(end - start))
    return local_max


def get_mask(h, w, levels, sigma):
    mask = np.zeros((h, w, levels))
    for i in range(levels):
        b = int(math.ceil(sigma[i] * math.sqrt(2)))  # Boundary.
        mask[b + 1:h - b, b + 1:w - b] = 1
    return mask


def non_maximum_suppression(scale_space, method):
    if method == 'rank':
        local_max = non_maximum_suppression_rank(scale_space)
    else:
        local_max = non_maximum_suppression_generic(scale_space)

    h = local_max.shape[0]
    w = local_max.shape[1]
    levels = local_max.shape[2]
    global_max = np.zeros(local_max.shape)

    for i in range(h):
        for j in range(w):
            max_value = np.amax(local_max[i, j, :])
            max_idx = np.argmax(local_max[i, j, :])
            global_max[i, j, max_idx] = max_value

    for i in range(levels):
        global_max[:, :, i] = np.where((global_max[:, :, i] ==
                                        scale_space[:, :, i]),
                                       global_max[:, :, i], 0)

    return global_max


def get_radius(sigma, num_rads):
    return np.ones(num_rads) * math.sqrt(2) * sigma


def detect_blob(global_max, threshold, sigma):
    levels = global_max.shape[2]

    mask = get_mask(global_max.shape[0],
                    global_max.shape[1],
                    levels, sigma)

    row_idx = []
    col_idx = []
    radius = []

    for i in range(levels):
        global_max[:, :, i] = np.where((global_max[:, :, i] > threshold) &
                                       (mask[:, :, i] == 1), 1, 0)
        row_idx.append(list(np.where(global_max[:, :, i] == 1)[0]))
        col_idx.append(list(np.where(global_max[:, :, i] == 1)[1]))
        radius.append(list(get_radius(sigma[i], len(row_idx[i]))))

    row_idx = list(chain.from_iterable(row_idx))
    col_idx = list(chain.from_iterable(col_idx))
    radius = list(chain.from_iterable(radius))

    return row_idx, col_idx, radius


def show_all_circles(save_path, image, cx, cy, rad, color='r'):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    for x, y, r in zip(cx, cy, rad):
        circ = Circle((x, y), r, color=color, fill=False)
        ax.add_patch(circ)
    plt.savefig(save_path, format="jpg")
    plt.title('%i circles' % len(cx))
    plt.show()


def run(dir, img_name, method, filter_name, levels, k, threshold, init_sigma):
    file_path = dir + img_name + ".jpg"
    img = get_gray_image(file_path)
    scale_space, sigma = get_scale_space(img, init_sigma, levels, k, method)
    global_max = non_maximum_suppression(scale_space, filter_name)
    row_idx, col_idx, radius = detect_blob(global_max, threshold, sigma)
    save_path = "result/" + img_name + "_" + method + "_" + filter_name + "_" + str(threshold) + ".jpg"
    show_all_circles(save_path, img, col_idx, row_idx, radius)


if __name__ == '__main__':
    dir = 'part2_images/'
    images = ['butterfly', 'einstein', 'fishes', 'sunflowers', 'car', 'bike', 'cat', 'dog']
    methods = ['increase', 'downsample']
    filters = ['rank', 'generic']
    levels = 12
    k = 1.3
    init_sigma = 2
    thresholds = np.array([0.00000001, 0.000000008])
    run(dir, images[7], methods[0], filters[0], levels, k, thresholds[0], init_sigma)
    run(dir, images[7], methods[1], filters[1], levels, k, thresholds[0], init_sigma)
