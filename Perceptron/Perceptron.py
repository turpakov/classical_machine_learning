import numpy as np
from itertools import groupby
import copy
import matplotlib.pyplot as plt


ALL_COLORS = ["red", "black", "green", "blue", "pink"]
ALL_MARKERS = [">", "*", "x", (5, 0), (5, 1)]


def init_data(filename: str):
    with open(filename, "r") as rf:
        input_data = []
        data = [s.strip() for s in rf.readlines()]
        for i in data:
            input_data.append([float(s) for s in i.split(',')])
    return input_data


def unificate(x):
    for data in x:
        for i in range(len(data) - 1):
            data[i] = data[-1] * data[i]
    return x


def heavyside(x):
    if all(k >= 0 for k in x):
        return 1
    else:
        return 0


def find_bord(x_for_print):
    if len(x_for_print[0]) == 3:
        x_min = float('inf')
        x_max = float('-inf')
        for x in x_for_print:
            if x[0] < x_min:
                x_min = x[0]
            if x[0] > x_max:
                x_max = x[0]
        return x_min, x_max
    if len(x_for_print[0]) == 4:
        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')
        for x in x_for_print:
            if x[0] < x_min:
                x_min = x[0]
            if x[0] > x_max:
                x_max = x[0]
            if x[1] < y_min:
                y_min = x[0]
            if x[1] > y_max:
                y_max = x[0]
        return x_min, x_max, y_min, y_max


def calc_line(x_for_print, w_0):
    if len(w_0) == 3:
        x_min, x_max = find_bord(x_for_print)
        y_for_print_min = (0 - w_0[0] * x_min - w_0[2]) / w_0[1]
        y_for_print_max = (0 - w_0[0] * x_max - w_0[2]) / w_0[1]
        return x_min, x_max, y_for_print_min, y_for_print_max
    if len(w_0) == 4:
        x_min, x_max, y_min, y_max = find_bord(x_for_print)
        f = lambda x, y: (0 - w_0[3] - x * w_0[0] - y * w_0[1]) / w_0[2]
        xval = np.linspace(x_min, x_max, 100)
        yval = np.linspace(y_min, y_max, 100)
        x, y = np.meshgrid(xval, yval)
        z = f(x, y)
        return x, y, z


def show_result(all_lines, x_for_print, classes, all_w):
    print(f"Классы: {classes}")
    print("Уравнения разделяющих гиперплоскостей:")
    for i, w in enumerate(all_w):
        s = f"{classes[i]}: "
        for j in range(len(w)):
            s += f"{w[j]} * x{j+1} + "
        print(s[:-8])
    if len(x_for_print[0]) == 3:
        plt.title('PERCEPTRON')
        for i in range(len(classes)):
            for x in x_for_print:
                if x[-1] == classes[i]:
                    plt.scatter(x[0], x[1], c=ALL_COLORS[i], marker=ALL_MARKERS[i])
        for i, line in enumerate(all_lines):
            plt.plot([line[0], line[1]], [line[2], line[3]], c=ALL_COLORS[i])
        plt.show()
    if len(x_for_print[0]) == 4:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(classes)):
            xdata = [x[0] for x in x_for_print if x[-1] == classes[i]]
            ydata = [x[1] for x in x_for_print if x[-1] == classes[i]]
            zdata = [x[2] for x in x_for_print if x[-1] == classes[i]]
            for j in range(len(xdata)):
                ax.scatter3D(xdata[j], ydata[j], zdata[j], c=ALL_COLORS[i], marker=ALL_MARKERS[i])
        for i, pl in enumerate(all_lines):
            x = pl[0]
            y = pl[1]
            z = pl[2]
            ax.plot_surface(x, y, z, color=ALL_COLORS[i])
        plt.title('PERCEPTRON')
        plt.show()


def error(w, V):
    err = w.dot(V.T)
    num_err = []
    e = 0
    for k in range(len(err)):
        if err[k] < 0:
            num_err.append(k + 1)
            e -= err[k]
    return e, num_err


def perceptron(input_data, w):
    x_for_print = copy.deepcopy(input_data)
    classes = sorted([x[-1] for x in input_data])
    classes = [x for x, _ in groupby(classes)]
    all_w = []
    all_lines = []
    for i in range(len(classes) - 1):
        x_for_calc = copy.deepcopy(input_data)
        for x in x_for_calc:
            if x[-1] == classes[i]:
                x[-1] = 1
            else:
                x[-1] = -1
        x_lst = np.array(x_for_calc)
        x_lst = unificate(x_lst)
        V = np.array(x_lst)
        iter = 0
        count = 100000
        cur_record_err = error(w, V)[0]
        record = w
        while iter < count:
            for j in range(len(V)):
                if w.dot(V[j].T) <= 0:
                    w = w + V[j]
            if error(w, V)[0] < cur_record_err:
                record = w
                cur_record_err = error(w, V)[0]
            iter += 1

        all_w.append(record)
        all_lines.append(calc_line(x_for_print, record))
        e, num_err = error(record, V)
        if num_err:
            print('Линейно неразделимы!')
            print(f'Относительно разделяющей прямой для класса {int(classes[i])} ошибочно было классифицировано точек: {len(num_err)}, а именно точки: {num_err}')
            print(f'Ошибка составила {e}')
    show_result(all_lines, x_for_print, classes, all_w)


input_data = init_data("test_multi_error.txt")

perceptron(input_data, np.array([0.5 for i in range(len(input_data[0]))]).T)

