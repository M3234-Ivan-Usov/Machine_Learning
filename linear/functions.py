from math import sqrt

import numpy as np


# NORMS
def l2_norm(vector):
    return sqrt(np.sum(np.fromiter(map(lambda x: x ** 2, vector), dtype=float)))


def l1_norm(vector):
    return np.sum(np.fromiter(map(lambda x: abs(x), vector), dtype=float))


# LOSS
def loss_function(model, item):
    f = model.features[item]
    w = model.weights
    expected = model.targets[item]
    actual = np.sum(f*w)
    return (actual - expected) ** 2


def loss_grad(model, item):
    f = model.features[item]
    w = model.weights
    expected = model.targets[item]
    actual = np.sum(f * w)
    return 2 * (actual - expected) / l2_norm(f) * f


# RIDGE
def ridge_function(model, item):
    return loss_function(model, item) + model.regularisation / 2 * l2_norm(model.weights) ** 2


def ridge_grad(model, item):
    return loss_grad(model, item) + model.regularisation * model.weights


# LASSO
def lasso_function(model, item):
    return loss_function(model, item) + model.regularisation * l1_norm(model.weights)


def lasso_grad(model, item):
    return loss_grad(model, item) + model.regularisation * sign(model.weights)


def sign(w):
    signed = np.empty(shape=len(w), dtype=float)
    for val in range(len(w)):
        if w[val] == 0.0:
            signed[val] = 0.0
            continue
        if w[val] > 0.0:
            signed[val] = 1.0
        else:
            signed[val] = -1.0
    return signed


def scalar_sign(x):
    if x > 0.0:
        return 1.0
    if x == 0.0:
        return 0.0
    return -1.0


def estimate_loss(model, loss_func):
    return np.sum(np.fromiter(map(lambda x: loss_func(model, x), range(model.data_amount)), dtype=float))
