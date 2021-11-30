import matplotlib.pyplot as plt
from itertools import groupby
import copy
import numpy as np



ALL_COLORS = ["red", "black", "green", "blue", "pink"]
ALL_MARKERS = [">", "*", "x", (5, 0), (5, 1)]


def step_func(z):
    if z > 0:
        return 1.0
    elif z < 0:
        return 0.0
    else:
        return 0.5


def find_bord(x_for_print):
    if len(x_for_print[0]) == 2:
        x_min = float('inf')
        x_max = float('-inf')
        for x in x_for_print:
            if x[0] < x_min:
                x_min = x[0]
            if x[0] > x_max:
                x_max = x[0]
        return x_min, x_max


def calc_line(x_for_print, w_0):
    if len(w_0) == 3:
        x1 = [min(x_for_print[:, 0]), max(x_for_print[:, 0])]
        m = -w_0[1] / w_0[2]
        c = -w_0[0] / w_0[2]
        x2 = m * x1 + c
        return x1, x2


def show_result(all_lines, x_for_print, y_for_print, classes, all_w):
    print(f"Классы: {classes}")
    print("Уравнения разделяющих гиперплоскостей:")
    for i, w in enumerate(all_w):
        s = f"{classes[i]}: "
        for j in range(len(w)):
            s += f"{w[j][0]} * x{j+1} + "
        print(s[:-8])
    if len(x_for_print[0]) == 2:
        for i in range(len(classes)):
            for j, x in enumerate(x_for_print):
                if y_for_print[j] == classes[i]:
                    plt.scatter(x[0], x[1], c=ALL_COLORS[i], marker=ALL_MARKERS[i])
        for i, line in enumerate(all_lines):
            plt.plot(line[0], line[1], c=ALL_COLORS[i])
        plt.show()


def perceptron(X, y, lr, epochs):
    # X --> Inputs.
    # y --> labels/target.
    # lr --> learning rate.
    # epochs --> Number of iterations.
    # m-> number of training examples
    # n-> number of features
    m, n = X.shape
    x_for_print = copy.deepcopy(X)
    y_for_print = copy.deepcopy(y)
    classes = sorted(y)
    classes = [x for x, _ in groupby(classes)]
    all_w = []
    all_lines = []
    for i in range(len(classes) - 1):
        w = np.zeros((n + 1, 1))
        x_for_calc = copy.deepcopy(X)
        y_for_calc = copy.deepcopy(y)
        for j in range(len(y_for_calc)):
            if y_for_calc[j] == classes[i]:
                y_for_calc[j] = 1
            else:
                y_for_calc[j] = 0
        for epoch in range(epochs):
            for idx, x_i in enumerate(x_for_calc):
                # Inserting 1 for bias, X0 = 1.
                x_i = np.insert(x_i, 0, 1).reshape(-1, 1)
                y_hat = step_func(np.dot(x_i.T, w))
                if y_hat - y_for_calc[idx] != 0:
                    w += lr * ((y_for_calc[idx] - y_hat) * x_i)
        err = []
        for idx, x_i in enumerate(x_for_calc):
            x_i = np.insert(x_i, 0, 1).reshape(-1, 1)
            y_hat = step_func(np.dot(x_i.T, w))
            err.append(y_hat - y_for_calc[idx])
        #err = [step_func(k) for k in np.array([np.append([1], x) for x in X]).dot(w).reshape(1, m)[0]] - y
        print(err)
        #print(sum([e**2 for e in err]) / 2)
        all_w.append(w)
        all_lines.append(calc_line(x_for_print, w))
    show_result(all_lines, x_for_print, y_for_print, classes, all_w)


# X, y = datasets.make_blobs(n_samples=100,n_features=2,
#                            centers=2,cluster_std=1.05,
#                            random_state=2)

def init_data(filename: str):
    with open(filename, "r") as rf:
        input_data = []
        data = [s.strip() for s in rf.readlines()]
        for i in data:
            input_data.append([float(s) for s in i.split(',')])
    return input_data

input_data = init_data("test3.txt")
X = np.array([x[:-1] for x in input_data])
y = np.array([x[-1] for x in input_data])

perceptron(X, y, 0.1, 1000)
