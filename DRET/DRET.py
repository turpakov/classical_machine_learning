import os
from types import new_class
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools

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

def DRET(classes,spheres):
    clasters_centers = dict()
    for i,j in classes.items():
        clasters_centers[i] = np.divide(sum(j),len(j))
        length_ = len(j[0]) #размерность
    #идея - смотрим расстояние от центра до каждой точки, находим максимальное - это и будет длина мин. радиус
    radiuses = dict()
    for key,value in classes.items():
        radiuses[key] = max([np.linalg.norm(el-clasters_centers[key]) for el in value])
    #по радиусу и центру можем построить сферу
    colors = ['r','g','black']
    #здесь достроить сферы и закинуть их центры в clasters_centers
    buff = []
    for j in itertools.combinations(classes.keys(),2):
        buff = []
        for el in classes[j[0]]: #для каждого объекта из одного класса, смотрим находится ли он внутри сферы другого класса
            if sum((el - clasters_centers[j[1]])**2) <= radiuses[j[1]]**2:
                buff.append(el)
        if len(buff) == 0:
            continue #нет элементов в пересечении
        if np.all(buff == classes[j[0]]) or np.all(buff == classes[j[1]]):
            continue #тогда одна сфера включает другую
        #если прошли сюда, то буфер содержит элементы из класса j[0], которые лежат в пересечении со сферой класса j[1]
        #выделим элементы из j[1], которые лежат в пересечении со сферой классса j[0]
        buff2 = []
        for el in classes[j[1]]:
            if sum((el - clasters_centers[j[0]])**2) <= radiuses[j[0]]**2:
                buff2.append(el)
        new_classes = dict()
        new_classes[j[0]] = buff
        new_classes[j[1]] = buff2
        spheres += DRET(new_classes,spheres) #  рекурсивно запускаем, чтобы достроить сферы, пока строятся
    #либо отрисовка круга, либо вывод уравнения сферы
    if length_ == 2:
        i = 0
        for key,value in clasters_centers.items():
            circle1 = plt.Circle((value[0],value[1]),radiuses[key],color=colors[i%3],fill=False)
            i += 1
            ax=plt.gca()
            ax.add_patch(circle1)
            plt.axis('scaled')
    j = 0
    for key,value in clasters_centers.items():
        res = ''
        for x in value:
            res += '(x' + str(j) + str((-1)*round(x,3)) + ')^2' + '+'
        res = res[:-1]
        res += '=' + str(round(radiuses[key],3))
        spheres.append(res)
    return spheres

if __name__ == '__main__':
    data = data_completition([],'real_data.txt')
    classes = dict()
    data_classes = dict()
    for el in data:
        if el[-1] not in classes.keys():
            classes[el[-1]] = []
        classes[el[-1]].append(np.array(el[:-1]))
    for el in classes.keys():
        classes[el] = np.array(classes[el])
    spheres = np.unique(DRET(classes,[]))
    print(spheres)
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    plt.scatter(x,y)
    plt.show()