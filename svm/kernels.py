from math import exp

import numpy as np


def linear(x, y, extra_arg=None):
    return np.sum(x * y)


def polynomial(x, y, deg):
    return (np.sum(x * y) + 1) ** deg


def gaussian(x, y, beta):
    return exp(-beta * l2_norm(x - y))


def l2_norm(vector):
    return np.sum(np.fromiter(map(lambda x: x ** 2, vector), dtype=float))
