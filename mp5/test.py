import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from skimage import color


def test():
    # file = 'moebius'
    file = 'tsukuba'
    if file == 'moebius':
        file1 = 'moebius1.png'
        file2 = 'moebius2.png'
    else:
        file1 = 'tsukuba1.jpg'
        file2 = 'tsukuba2.jpg'
    img = io.imread('data/{}'.format(file1))
    plt.imshow(img)
    plt.show()
    gray_img = color.rgb2gray(img)
    gray_img /= 255
    plt.figure()
    plt.imshow(gray_img)
    plt.show()
    im1 = cv2.imread('data/{}'.format(file1))
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1 = im1.astype(float)
    im2 = cv2.imread('data/{}'.format(file2))
    im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    im2 = im2.astype(float)

    print(im1)
    plt.figure()
    plt.imshow(im1)
    plt.show()
    plt.savefig('test.jpg')
    # height, width = im1.shape
    # print(height, width)


if __name__ == '__main__':
    test()
