import os
import numpy as np

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


def calculate_E(f,U,num_class):
    positive = negative = 0
    for x in U:
        if x[-1] == num_class:
            positive += f(x)
        else:
            negative += f(x)
    if positive + negative == 0:
        return 1
    return negative/(positive+negative)


def DL(data,F,T_max,I_min,E_max,l_0,Y):
    U = data
    res = []
    for t in range(T_max):
        c = Y[t]
        decisive_rules_new = []
        for f in F:
            if calculate_E(f,U,c) <= E_max:
                decisive_rules_new.append(f)
        ind = np.argmax([calculate_I(f,U,c) for f in decisive_rules_new])
        f = decisive_rules_new[ind]
        if calculate_I(f,U,c) < I_min:
            break
        res.append([f,c])
        for i in reversed(range(len(U))):
            if f(U[i]) == 1:
                del U[i]
        if len(U) <= l_0:
            break
    return res

if __name__ == '__main__':
    data = data_completition([],'real_data_big.txt')
    np.random.shuffle(data)
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
    Y = [0,1,2]
    print(DL(data,F,T_max,I_min,E_max,l_0,Y))