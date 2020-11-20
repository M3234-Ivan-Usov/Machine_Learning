import numpy as np

import linear.estimation as estimation
from linear.functions import estimate_loss
from linear.structures import Model


def compute(dataset, loss_function):
    print("--- Starting least squares ---")
    approximations = list()
    regularisation = estimation.initialise_regularisation()
    while estimation.has_next(regularisation):
        pseudo_inverse = reverse_Moore_Penrose(dataset, regularisation)
        weights = pseudo_inverse.dot(dataset.targets)
        model = Model(dataset, weights, regularisation, 0.0, 0.0)
        error = estimate_loss(model, loss_function)
        approximations.append((weights, regularisation, error))
        regularisation = estimation.next_coefficient(regularisation)
    approximations.sort(key=lambda x: x[2])
    print("--- Least squares finished ---\n")
    return approximations[0]


def reverse_Moore_Penrose(dataset, regularisation):
    amount = dataset.features_amount
    reg_matrix = np.zeros(shape=(amount, amount), dtype=float)
    matrix = dataset.features
    for i in range(amount):
        reg_matrix[i, i] = regularisation
    return np.linalg.inv(matrix.transpose().dot(matrix) + reg_matrix).dot(matrix.transpose())
