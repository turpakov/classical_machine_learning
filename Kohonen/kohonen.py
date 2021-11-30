import copy
import matplotlib.pyplot as plt
import numpy as np
import random


def kohonen(n_classes, x_values, dim, h):
    cores = np.array([x_values[random.randint(0, len(x_values))-1] for i in range(n_classes)])
    matrix_distance = np.array([[np.linalg.norm(x - core) for core in cores] for x in x_values])

    while True:
        prev_cores = copy.deepcopy(cores)
        for i, dist in enumerate(matrix_distance):
            cores[np.argmin(dist)] += h * (x_values[i] - cores[np.argmin(dist)])
        matrix_distance = np.array([[np.linalg.norm(x - core) for core in cores] for x in x_values])
        if (cores == prev_cores).all():
            img1 = x_values.reshape(dim, dim, -1)
            plt.figure(1)
            plt.subplot(121)
            for i, d in enumerate(matrix_distance):
                col = i // dim
                row = i % dim
                plt.text(row, col, f'{np.argmin(d)+1}', horizontalalignment='center', verticalalignment='center')
            plt.imshow(img1)
            break

    plt.subplot(122)
    for i in range(n_classes):
        plt.text(0, i, i+1, horizontalalignment='center', verticalalignment='center')
    plt.imshow(cores.reshape(n_classes, 1, -1))
    plt.show()


n_classes = 10
dim = 7
x_values = np.array([np.random.uniform(0, 1, 3) for i in range(dim * dim)])
h = 0.1
kohonen(n_classes, x_values, dim, h)
