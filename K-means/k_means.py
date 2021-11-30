from sklearn.metrics import pairwise_distances_argmin, pairwise_distances
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()


def find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        labels = pairwise_distances_argmin(X, centers)
        new_centers = np.array([X[labels==i].mean(axis=0) for i in range(n_clusters)])
        if np.all(centers == new_centers):
            break
        centers = new_centers

    sum_squad_err = sum([np.linalg.norm(centers[i] - j) ** 2 for i in range(n_clusters) for j in X[labels==i]])
    print(f"{sum_squad_err} - средняя квадратичная ошибка в моей для числа классов {n_clusters}")
    return centers, labels, sum_squad_err

all_sse = []
all_res = []
k_rng = range(1, 7)
for n_cluster in k_rng:
    centers, labels, sum_squad_err = find_clusters(X, n_cluster)
    all_res.append(labels)
    all_sse.append(sum_squad_err)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, all_sse)
plt.show()

res = all_res[3]
plt.scatter(X[:, 0], X[:, 1], c=res, s=50, cmap='viridis')
plt.title("My KMeans")
plt.show()

sum_diam = 0
for label in range(4):
    cur_rec = 0
    cur_x = X[res==label]
    dist_matr = pairwise_distances(cur_x, cur_x)
    for d in dist_matr:
        if max(d) > cur_rec:
            cur_rec = max(d)
    sum_diam += cur_rec
print(f"{sum_diam / 4} - средняя длина диаметров в моем")

km = KMeans(n_clusters=4, random_state=0)
y_pred = km.fit_predict(X)
sum_diam = 0
for label in range(4):
    cur_rec = 0
    cur_x = X[y_pred==label]
    dist_matr = pairwise_distances(cur_x, cur_x)
    for d in dist_matr:
        if max(d) > cur_rec:
            cur_rec = max(d)
    sum_diam += cur_rec
print(f"{km.inertia_} - средняя квадратичная ошибка в sklearn")
print(f"{sum_diam / 4} - средняя длина диаметров в sklearn")
plt.subplot(111)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
plt.title("KMeans from sklearn")
plt.show()

