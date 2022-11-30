import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

# file = 'moebius'
file = 'tsukuba'
if file == 'moebius':
    file1 = 'moebius1.png'
    file2 = 'moebius2.png'
else:
    file1 = 'tsukuba1.jpg'
    file2 = 'tsukuba2.jpg'

file_name1 = 'data/{}'.format(file1)
file_name2 = 'data/{}'.format(file2)
input_left = cv2.imread(file_name1, 0)
input_right = cv2.imread(file_name2, 0)

imgHeight, imgWidth = input_left.shape
print(input_left.shape)
print(input_right.shape)

plt.imshow(input_left, cmap='gray_r')

t1 = time.time_ns()

DbasicSubpixel = np.zeros(input_left.shape)

disparityRange = 1
halfBlockSize = 2
blockSize = 2 * halfBlockSize + 1
method = 'sad'

for m in range(imgHeight):
    minr = max(0, m - halfBlockSize)
    maxr = min(imgHeight - 1, m + halfBlockSize)
    for n in range(imgWidth):
        minc = max(0, n - halfBlockSize)
        maxc = min(imgWidth - 1, n + halfBlockSize)

        mind = max(-disparityRange, -minc)
        # mind = 0
        maxd = min(disparityRange, imgWidth - 1 - maxc)

        template = input_left[minr:maxr + 1, minc:maxc + 1]
        numBlocks = maxd - mind + 1
        blockDiffs = np.zeros(numBlocks)

        ncc = 0
        nccNumerator = 0
        nccDenominator = 0
        nccDenominatorRightWindow = 0
        nccDenominatorLeftWindow = 0

        for i in range(mind, maxd + 1):
            block = input_right[minr:maxr + 1, minc + i:maxc + i + 1]
            blockIndex = i - mind
            for j in range(minr, maxr + 1):
                for k in range(minc, maxc + 1):
                    # SAD
                    if method == 'sad':
                        blockDiff = np.abs(input_right[j, k] - input_left[j, k + i])
                        blockDiffs[blockIndex] = blockDiffs[blockIndex] + blockDiff
                    elif method == 'ssd':
                        blockDiff = np.sqrt(np.abs(input_right[j, k] - input_left[j, k + i]))
                        blockDiffs[blockIndex] = blockDiffs[blockIndex] + blockDiff
                    else:
                        nccNumerator = nccNumerator + (input_right[j, k] * input_left[j, k + i])
                        nccDenominatorLeftWindow = nccDenominatorLeftWindow + (
                                input_left[j, k + i] * input_left[j, k + i])
                        nccDenominatorRightWindow = nccDenominatorRightWindow + (input_right[j, k] * input_right[j, k])

            # nccDenominator = np.sqrt(nccDenominatorRightWindow * nccDenominatorLeftWindow)
            # ncc = nccNumerator / nccDenominator
            # blockDiffs[blockIndex, 1] = ncc
        if method == 'ncc':
            bestMatchIndex = np.argmax(np.array(blockDiffs))
        else:
            bestMatchIndex = np.argmin(np.array(blockDiffs))
        d = bestMatchIndex + mind

        if (bestMatchIndex == 0) or (bestMatchIndex == numBlocks - 1):
            DbasicSubpixel[m, n] = d
        else:
            C1 = blockDiffs[bestMatchIndex - 1]
            C2 = blockDiffs[bestMatchIndex]
            C3 = blockDiffs[bestMatchIndex + 1]
            DbasicSubpixel[m, n] = d - (0.5 * (C3 - C1) / (C1 - (2 * C2) + C3))

    if np.mod(m, 10) == 0:
        print('图像行：{} / {} ({})\n'.format(m, imgHeight, (m / imgHeight) * 100))

t2 = time.time_ns()
print("time = {}".format(t2 - t1))
plt.imshow(DbasicSubpixel, cmap='gray_r')

import scipy.signal as signal

DbasicSubpixel_2 = signal.medfilt2d(DbasicSubpixel, (5, 5))
plt.imshow(DbasicSubpixel_2, cmap='gray_r')
