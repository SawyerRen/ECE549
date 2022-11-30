import numpy as np

D = np.loadtxt('factorization_data/measurement_matrix.txt')
m, n = D.shape

print(m)
print(n)

norm_D = np.zeros((m, n))
for i in range(m):
    norm_D[i, :] = D[i, :] - np.mean(D[i, :])

import scipy

U, W, V = np.linalg.svd(norm_D, full_matrices=False)

W = np.diag(W)
print(U.shape)
print(W.shape)
print(V.shape)
U3 = U[:, 0:3]
W3 = W[0:3, 0:3]
# V3 = V[:, 0:3]
V3 = V[0:3, :]

M = np.dot(U3, scipy.linalg.fractional_matrix_power(W3, 0.5))
S = np.dot(scipy.linalg.fractional_matrix_power(W3, 0.5), V3)
estimated_D = np.dot(M, S)
# estimated_D = np.zeros((m, n))
for i in range(m):
    estimated_D[i, :] = estimated_D[i, :] + np.mean(D[i, :])
print(estimated_D.shape)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(S[0, :], S[1, :], S[2, :])
ax.view_init(160, 160)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

from PIL import Image

frames = [1, 2, 3]
for i in range(3):
    index = frames[i]
    file_name = 'factorization_data/frame0000000{}.jpg'.format(index)
    I = Image.open(file_name)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(I, cmap='gray_r')
    ax.scatter(D[2 * index - 1, :], D[2 * index, :], c='r')
    # ax.scatter(estimated_D[2 * index - 1, :] + np.mean(D[2 * index - 1, :]),
    #            estimated_D[2 * index, :] + np.mean(D[2 * index, :]),c='none',marker='o',edgecolors='g')
    ax.scatter(estimated_D[2 * index - 1, :],
               estimated_D[2 * index, :], c='none', marker='o', edgecolors='g')
    plt.show()

print(m)
residual = np.zeros((101, 1))
for i in range(m):
    for j in range(n):
        residual[int(i / 2), 0] += np.sqrt(abs(estimated_D[i, j] - D[i, j]))
print(residual)
plt.plot(residual)
plt.xlabel('frames')
plt.ylabel('residual')
plt.show()
