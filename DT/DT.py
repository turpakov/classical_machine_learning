import os
import numpy as np
from numpy.core.fromnumeric import argmax, argmin

def data_completition(V,filename):
    with open(os.path.join(os.getcwd(),filename),'r',encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split(',')
            time = []
            for item in line:
                if '(' in item:
                    time += [float(item[1:])]
                if ')' in item:
                    time += [float(item[:item.index(')')])]
                elif '(' not in item and ')' not in item:
                    time += [float(item)]
            V.append(time)
    return V

def f1(x): #легкие танки
    return int(x[0] < 20)

def f2(x): #быстрые танки
    return int(x[4]  > 50)

def f3(x): #тяжелые танки
    return int(x[0] > 40)

def f4(x): #легкие и быстрые танки
    return int(x[0] < 20) * int(x[4]  > 50)

def f5(x):#тяжелые и быстрые танки
    return int(x[0] > 40) * int(x[4]  > 50)

def decisive_rules():
    return [f1,f2,f3,f4,f5]

def C(n, k):
    if 0 <= k <= n:
        nn = 1
        kk = 1
        for t in range(1, min(k, n - k) + 1):
            nn *= n
            kk *= t
            n -= 1
        return nn // kk
    else:
        return 0

def calculate_I(f,U,num_class):
    positive = negative = P = N =  0
    for x in U:
        if x[-1] == num_class:
            positive += f(x)
            P += 1
        else:
            negative += f(x)
            N += 1
    return (-1) * np.log(C(P,positive)*C(N,negative)/C(P+N,positive+negative))

class Tree:
    def __init__(self):
        # self.parent = None
        self.left_branch = None
        self.right_branch = None
        self.class_ = None
        self.f = None


def LearnID3(U):
    buff = np.unique(U[:,-1])
    if len(buff) == 1:
        v = Tree()
        v.class_ = buff[0]
        # v.parent = root
        return v
    b = []
    for el in buff:
        for f in decisive_rules():
            b.append(calculate_I(f,U,el))
    ind = argmax(b) % len(decisive_rules())
    f = decisive_rules()[ind]
    U_0 = []
    U_1 = []
    for x in U:
        if f(x) == 0:
            U_0.append(x)
        else:
            U_1.append(x)
    if len(U_0)*len(U_1) == 0:
        v_new = Tree()
        ind = argmax([[i for i in U[:,1]].count(el) for el in buff])
        v_new.class_ = buff[ind]
        # v_new.parent = root
    else:
        v_new = Tree()
        # v_new.parent = root
        v_new.f = f
        v_new.left_branch = LearnID3(np.array(U_0))
        v_new.right_branch = LearnID3(np.array(U_1))
    return v_new

def DT(data):
    U = data
    v = LearnID3(U)
    return v
    
def predict(v,test):
    while True:
        buff = v.f(test)
        if buff == 0:
            v = v.left_branch
        else:
            v = v.right_branch
        class_ = v.class_
        if v.left_branch is None:
            return class_


if __name__ == '__main__':
    data = np.array(data_completition([],'real_data_big.txt'))
    # np.random.shuffle(data)
    # num_training_data = int(0.8 * len(data))
    # data_train = data[:num_training_data]
    # data_test = data[num_training_data:]
    T_max = 3
    I_min = 2.5
    E_max = 0.3
    l_0 = 1
    F = decisive_rules()
    #Решающие правила
    #легкие и быстрые танки - 1й класс
    #тяжелые и быстрые танки - 2й класс
    test = data[2]
    v = DT(data)
    print(predict(v,test))