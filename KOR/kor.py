import pandas as pd
from itertools import combinations_with_replacement, zip_longest, groupby
import json
import numpy as np


def getBin(number, total):
    return bin(number)[2:].zfill(total)


ALL_BIN = {
    2: [list(map(int, getBin(i, 2))) for i in range(2 ** 2)],
    3: [list(map(int, getBin(i, 3))) for i in range(2 ** 3)],
    4: [list(map(int, getBin(i, 4))) for i in range(2 ** 4)]
}
TAU = 4
ALL_CLASSES = [1, 2]
MAX_CON_LEN = 4


def kor(data):
    result = []
    new_result = []
    # фиксируем класс
    for cls in ALL_CLASSES:
        # длина конъюнкции
        for len_con in range(2, MAX_CON_LEN + 1):
            # номера столбцов
            for con in combinations_with_replacement(list(range(0, 16)), len_con):
                # попарно различные признаки
                if len(list(groupby(sorted(con)))) != len(con):
                    continue
                # все возможные варианты событий (все бинарные вектора длины len_con)
                all_bin = ALL_BIN[len_con]
                for bin in all_bin:
                    # текущая вырезка столбцов данных текущего класса
                    rows = [data[i] for i in con]
                    cur_slice_from_data = zip_longest(*rows)
                    count = 0
                    is_clear = 1
                    for i, row in enumerate(cur_slice_from_data):
                        if tuple(bin) == row:
                            count += 1
                            if data[-1][i] != cls:
                                is_clear = 0
                                break
                    # проверка на непротиворечивость и минимальную частоту встречаемости
                    if is_clear and count >= TAU:
                        result.append({
                            'class': cls,
                            'len_conunction': len_con,
                            'column_num': con,
                            'count': count,
                            'bit_mask': bin,
                        })
    # удалим дочерние
    bad_ids = []
    for cls in ALL_CLASSES:
        cur_res = [data for data in result if data['class'] == cls]
        for i in range(len(cur_res)):
            for j in range(i + 1, len(cur_res)):
                if set(cur_res[i]['column_num']).issubset(cur_res[j]['column_num']):
                    count_match = 0
                    for id, b in enumerate(cur_res[i]['bit_mask']):
                        if b == cur_res[j]['bit_mask'][cur_res[j]['column_num'].index(cur_res[i]['column_num'][id])]:
                            count_match += 1
                    if count_match == len(cur_res[i]['column_num']):
                        bad_ids.append(j)
    for i in range(len(result)):
        if i not in bad_ids:
            new_result.append(result[i])
    with open('result.json', 'w') as wf:
        json.dump(new_result, wf, indent=4)
    return new_result


df = pd.read_csv('kor_dataset.csv', encoding='windows-1251', sep=';')
d = df.to_numpy()
np.random.shuffle(d)
data = d.T
tests = np.array([[0, 1 ,0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                  [0, 1 ,1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0]])

result = kor(data)

for test in tests:
    voices = 0
    for res in result:
        if [test[i] for i in res['column_num']] == res['bit_mask'] and res['class'] == 1:
            voices += 1
        if [test[i] for i in res['column_num']] == res['bit_mask'] and res['class'] == 2:
            voices -= 1
    if voices > 0:
        print(f'{test} - Патриот')
    else:
        print(f'{test} - Демократ')
