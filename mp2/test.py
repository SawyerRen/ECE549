# -*- coding = utf-8 -*-
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


def load_and_preprocess(path):
    # Load image.
    img = io.imread(path)
    # Gray and scale.
    img = color.rgb2gray(img)
    img /= 255
    return img


# Get squared Laplacian response in scale-space.
def get_scale_space(image, init_sigma, levels, k, method):
    sigma = np.zeros(levels)
    sigma[0] = init_sigma

    h = image.shape[0]
    w = image.shape[1]

    scale_space = np.empty((h, w, levels))

    # Method 1. (slower version)
    def increase_filter_size():
        for i in range(levels):
            # Ensure odd filter size.
            filter_size = int(round(6 * sigma[i]))
            if filter_size % 2 == 0:
                filter_size += 1

            # Initialize filter matrix.
            center = int((filter_size + 1) / 2 - 1)  # index of center entry
            gauss_filter = np.zeros((filter_size, filter_size))
            gauss_filter[center][center] = 1

            # Obtain filter with normalization and apply convolution.
            LoG = sigma[i] * sigma[i] * gaussian_laplace(gauss_filter, sigma[i])
            scale_space[:, :, i] = convolve(image, LoG) ** 2

            # Update sigma.
            if (i + 1 < levels):
                sigma[i + 1] = sigma[i] * k

    # Method 2. (faster version)
    def downsample():
        # Ensure odd filter size.
        filter_size = int(round(6 * sigma[0]))
        if filter_size % 2 == 0:
            filter_size += 1

        # Initialize filter matrix.
        center = int((filter_size + 1) / 2 - 1)  # index of center entry
        gauss_filter = np.zeros((filter_size, filter_size))
        gauss_filter[center][center] = 1

        # Obtain filter (no normalization needed).
        LoG = gaussian_laplace(gauss_filter, sigma[0])

        # Scale the image.
        for i in range(levels):
            # Down scale.
            scaled_h = int(h * ((1 / k) ** i))
            scaled_w = int(w * ((1 / k) ** i))
            # scaled_im = transform.resize(image, (scaled_h, scaled_w), order=3)
            scaled_im = transform.rescale(image, (1 / k) ** i, order=3)

            # Apply convolution without normalization.
            im_tmp = convolve(scaled_im, LoG) ** 2

            # Upscale.
            scale_space[:, :, i] = transform.resize(im_tmp, (h, w), order=3)

            # Update sigma.
            if (i + 1 < levels):
                sigma[i + 1] = sigma[i] * k

    # Run method and record time.
    print('Calculating squared Laplacian response in scale-space...')

    start = time.time()
    run = {'increase_filter_size': increase_filter_size,
           'downsample': downsample}
    run[method]()
    end = time.time()

    print('Method used: ' + method +
          '; Time elapsed: {} s.'.format(end - start))

    return scale_space, sigma


# Helper func: non-maximum suppression in each 2D slice.
def non_max_sup_2D(scale_space, method):
    h = scale_space.shape[0]
    w = scale_space.shape[1]
    levels = scale_space.shape[2]

    local_max = np.empty((h, w, levels))

    # Method 1: rank_filter.
    # This method is much faster. Use this.
    def rank():
        for i in range(levels):
            curr_response = scale_space[:, :, i]
            local_max[:, :, i] = rank_filter(curr_response, -1, (3, 3))

    # Method 2: generic_filter.
    def generic():
        for i in range(levels):
            curr_response = scale_space[:, :, i]

            find_max = lambda arr: np.amax(arr)
            local_max[:, :, i] = generic_filter(curr_response, find_max, (3, 3))

    # Run method and record time.
    print('Running local non-max suppression...')

    start = time.time()
    run = {'rank': rank, 'generic': generic}
    run[method]()
    end = time.time()

    print('Method used: ' + method +
          '; Time elapsed: {} s.'.format(end - start))

    return local_max


# Helper func: compute radius of each local maximum.
def get_radius(sigma, num_rads):
    return np.ones(num_rads) * math.sqrt(2) * sigma


# Helper func: mask filter to eliminate boundaries noises.
def get_mask(h, w, levels, sigma):
    mask = np.zeros((h, w, levels))
    for i in range(levels):
        b = int(math.ceil(sigma[i] * math.sqrt(2)))  # Boundary.
        mask[b + 1:h - b, b + 1:w - b] = 1
    return mask


# Non-maximum suppression in 3D scale space.
def non_max_sup_3D(scale_space, sigma):
    # Obtain local 2D non max sup using rank_filter.
    local_max = non_max_sup_2D(scale_space, 'rank')

    h = local_max.shape[0]
    w = local_max.shape[1]
    levels = local_max.shape[2]

    # Compute non-max suppression accorss all layers.
    print("Running global non-max suppression...")
    global_max = np.zeros(local_max.shape)

    for i in range(h):
        for j in range(w):
            max_value = np.amax(local_max[i, j, :])
            max_idx = np.argmax(local_max[i, j, :])
            global_max[i, j, max_idx] = max_value

    # Eliminate duplicate values.
    for i in range(levels):
        global_max[:, :, i] = np.where((global_max[:, :, i] ==
                                        scale_space[:, :, i]),
                                       global_max[:, :, i], 0)

    print("Done with global non-max suppression.")
    return global_max


# Obtain center points and radius of blobs.
def detect_blob(global_max, threshold, sigma):
    levels = global_max.shape[2]

    mask = get_mask(global_max.shape[0],
                    global_max.shape[1],
                    levels, sigma)

    row_idx = []
    col_idx = []
    radius = []

    print('Finding blobs...')
    for i in range(levels):
        global_max[:, :, i] = np.where((global_max[:, :, i] > threshold) &
                                       (mask[:, :, i] == 1), 1, 0)

        # Obtain row & column index for local maxima.
        row_idx.append(list(np.where(global_max[:, :, i] == 1)[0]))
        col_idx.append(list(np.where(global_max[:, :, i] == 1)[1]))

        # Compute radius.
        radius.append(list(get_radius(sigma[i], len(row_idx[i]))))

    # Flatten nested list.
    row_idx = list(chain.from_iterable(row_idx))
    col_idx = list(chain.from_iterable(col_idx))
    radius = list(chain.from_iterable(radius))

    print('Done with finding blobs.')
    return row_idx, col_idx, radius


def show_and_save_all_circles(image, cx, cy, rad,
                              root, img_name, method, threshold, color='r'):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    for x, y, r in zip(cx, cy, rad):
        circ = Circle((x, y), r, color=color, fill=False)
        ax.add_patch(circ)

    plt.title('%i circles' % len(cx))

    plt.savefig(root + img_name + '_'
                + method + '_' + str(threshold) + '.svg', format='svg')

    plt.show()


# Pipeline for running the whole program.
def run_detection(root, img_name, method, levels, k, threshold, init_sigma):
    # Load and preprocess image.
    path = root + img_name + '.jpg'
    img = load_and_preprocess(path)

    # Get response.
    scale_space, sigma = get_scale_space(img, init_sigma, levels, k, method)

    # Non-max suppression.
    global_max = non_max_sup_3D(scale_space, sigma)

    # Get blobs.
    row_idx, col_idx, radius = detect_blob(global_max, threshold, sigma)

    # Display and save output.
    show_and_save_all_circles(img, col_idx, row_idx, radius,
                              root, img_name, method, threshold)


if __name__ == '__main__':
    root = 'part2_images/'

    # Choose one image.
    images = ['butterfly', 'einstein', 'fishes', 'sunflowers']
    # Choose one method.
    methods = ['increase_filter_size', 'downsample']

    # Configure parameters.
    levels = 12  # Scale levels
    k = 1.25  # Scale factor
    init_sigma = 2

    # Good thresholds for method 0 and 1.
    thresholds = np.array([0.0000001, 0.000000008])

    run_detection(root, images[0], methods[0], levels, k, thresholds[0], init_sigma)