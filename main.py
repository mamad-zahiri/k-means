#!/usr/bin/env python3

import numpy as np


def distance(center: np.array, point: np.array):
    return ((center - point) ** 2).sum()


def find_cluster(centers, point):
    dist = distance(centers[0], point)
    # nearest center index in "centers"
    index = 0
    for x in range(1, len(centers)):
        if distance(centers[x], point) < dist:
            index = x
    return index


def classify(points: np.array, k: int):
    number, _ = points.shape

    clusters = np.zeros((number, k), dtype=np.bool8)
    centers = points[np.random.randint(0, number, k)].astype(np.float32)

    old_centers = None
    centers_diff = 1

    EPSILON = 1e-10

    while centers_diff > EPSILON:
        clusters.fill(False)
        for point in range(number):
            cluster = find_cluster(centers, points[point])

            clusters[point, cluster] = True

        old_centers = centers.copy()
        for x in range(k):
            centers[x] = points[clusters[:, x]].mean(axis=0)
        centers_diff = (np.abs((centers - old_centers) ** 2).sum(axis=1)).max()

    return clusters.sum(axis=0)


if __name__ == "__main__":
    import sys

    iris = np.genfromtxt("iris.csv", delimiter=",")
    k = int(sys.argv[1])

    print(classify(iris, k))
