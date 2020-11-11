import math
from math import sqrt
from math import exp


def uniform(u):
    return 1.0 / 2.0 if u < 1 else 0.0


def triangular(u):
    return 1 - abs(u) if u < 1 else 0.0


def epanenchikov(u):
    return (1 - u * u) * 3 / 4 if u < 1 else 0.0


def quartic(u):
    return 15 / 16 * (1 - u * u) ** 2 if u < 1 else 0.0


def gaussian(u):
    return 1 / sqrt(2 * math.pi) * exp(-u * u / 2) if u < 1 else 0.0


def kernel(index):
    kernels = [uniform, triangular, epanenchikov, quartic, gaussian]
    return kernels[index]


def euclidean(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


def manhattan(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += abs(row1[i] - row2[i])
    return distance


def chebyshev(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        x = abs(row1[i] - row2[i])
        distance = max(x, distance)
    return distance


def distance(index):
    distances = [euclidean, manhattan, chebyshev]
    return distances[index]
