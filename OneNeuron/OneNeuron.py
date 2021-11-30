import numpy as np


def sign(x):
    if x.__class__.__name__ != 'ndarray':
        return 1 if x >= 0 else -1
    return np.array([1 if i >= 0 else -1 for i in x])


def init_data(filename:str):
    with open(filename, "r") as rf:
        input_data = []
        data = [s.strip() for s in rf.readlines()]
        for i in data:
            input_data.append([int(s) for s in i.split(',')])
    X = []
    Y = []
    for data in input_data:
        X.append(data[:2])
        Y.append(data[2])
    X = np.array(X)
    Y = np.array(Y)
    return (X, Y)


class Neuron:
    def __init__(self, init_weight):
        self.w = init_weight

    def fit(self, X, Y):
        self.x = X
        self.y = Y
        while True:
            for i in range(len(self.x)):
                self.w = self.w + self.x[i] * self.y[i]
            s = sign(self.x.dot(self.w))
            if s.all() == self.y.all():
                print(f"Итоговые веса: {self.w}")
                break

    def predict(self, X):
        ans = sign(X.dot(self.w))
        if ans >= 0:
            print(f"Вектор {X} принадлежит к классу 1")
        else:
            print(f"Вектор {X} принадлежит к классу 2")


X, Y = init_data("test2.txt")
nn = Neuron(np.array([0, 0]))
nn.fit(X, Y)
nn.predict(np.array([0, -2]))
nn.predict(np.array([0, -1]))
nn.predict(np.array([-100, -0.1]))
nn.predict(np.array([100, 10]))
nn.predict(np.array([-10, 0.5]))
