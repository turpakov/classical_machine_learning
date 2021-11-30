from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin



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


def replaceWithCentroid(labels, centers):
    new_pixels = []
    for label in labels:
        pixel_as_centroid = list(centers[label])
        new_pixels.append(pixel_as_centroid)
    new_pixels = np.array(new_pixels).reshape(*ori_img.size, -1)
    return new_pixels


def plotImage(img_array, size):
    return np.array(img_array / 255).reshape(*size)


ori_img = Image.open("lena.png")
X = np.array(ori_img.getdata())
n_clusters = 2
kmeans_my = find_clusters(X, n_clusters)
kmeans_sk = KMeans(n_clusters=n_clusters).fit(X)
print(f"{kmeans_sk.inertia_}  - средняя квадратичная ошибка в sklearn для числа классов {n_clusters}")
labels_my = kmeans_my[1]
centers_my = kmeans_my[0]
new_pixels_my = replaceWithCentroid(labels_my, centers_my)
labels_sk = kmeans_sk.labels_
centers_sk = kmeans_sk.cluster_centers_
new_pixels_sk = replaceWithCentroid(labels_sk, centers_sk)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
my = plotImage(new_pixels_my, new_pixels_my.shape)
ax1.imshow(my)
ax1.set_title('My')
ax1.axis("off")
sk = plotImage(new_pixels_sk, new_pixels_sk.shape)
ax2.imshow(sk)
ax2.set_title('Sklearn')
ax2.axis("off")
orig = X.reshape(*ori_img.size, -1)
ax3.imshow(orig)
ax3.set_title('Original')
ax3.axis("off")
plt.show()
