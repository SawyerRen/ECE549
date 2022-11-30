import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from skimage import color


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def gray():
    # file = 'moebius'
    file = 'tsukuba'
    if file == 'moebius':
        file1 = 'moebius1.png'
        file2 = 'moebius2.png'
    else:
        file1 = 'tsukuba1.jpg'
        file2 = 'tsukuba2.jpg'
    img = Image.open('data/{}'.format(file1))

    img.convert("1")
    img.show()
    im1 = cv2.imread('data/{}'.format(file1))
    # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # im1 = im1.astype(float)
    # im2 = cv2.imread('data/{}'.format(file2))
    # im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    # im2 = im2.astype(float)
    im1 = rgb2gray(im1)
    print(im1)
    plt.figure()
    plt.imshow(im1)
    plt.show()
    plt.savefig('test.jpg')
    # height, width = im1.shape
    # print(height, width)


def gray1():
    img = cv2.imread('data/tsukuba1.jpg', 0)

    # cv2.imwrite('res.jpg', img)


if __name__ == '__main__':
    gray1()
