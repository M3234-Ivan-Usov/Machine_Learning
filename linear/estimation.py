from math import sqrt

import numpy as np
from matplotlib import pyplot as plt
from structures import Model
import gradient

# Do not know, how to initialise
REG_MIN = 0.0
REG_MAX = 1.0
REG_STEP = 0.5


def initialise_regularisation():
    return REG_MIN


def next_coefficient(reg):
    return reg + REG_STEP


def has_next(reg):
    return reg <= REG_MAX


def smape(dataset, weights):
    total = 0.0
    for row in range(dataset.data_amount):
        expected = dataset.targets[row]
        actual = np.sum(dataset.features[row] * weights)
        total += 2 * abs(actual - expected) / (abs(actual) + abs(expected))
    return total / dataset.data_amount


def nrmse(dataset, weights):
    total = 0.0
    for row in range(dataset.data_amount):
        expected = dataset.targets[row]
        actual = np.sum(dataset.features[row] * weights)
        total += (actual - expected) ** 2
    rmse = sqrt(total / dataset.data_amount)
    return rmse / get_quartiles(dataset.targets)


def get_quartiles(targets):
    arranged_targets = np.sort(targets)
    values_amount = len(arranged_targets)
    quartile1 = arranged_targets[values_amount // 4]
    quartile3 = arranged_targets[3 * values_amount // 4]
    return quartile3 - quartile1


def estimate(dataset, approximation):
    smape_estimation = smape(dataset, approximation[0])
    nrmse_estimation = nrmse(dataset, approximation[0])
    print("Best regularisation: %s" % approximation[1])
    print("SMAPE: %s" % smape_estimation)
    print("NRMSE: %s\n" % nrmse_estimation)


def draw_plot(dataset, regularisation, grad_type):
    loss_function, grad_function, grad_step, iterations = gradient.configure(grad_type, dataset.targets)
    initial_step = 1 / get_quartiles(dataset.targets)
    weights = gradient.initialise_weights(dataset)
    model = Model(dataset, weights, regularisation, initial_step, gradient.EPS + 1)
    progress = [5, iterations // 20]
    losses = list()
    print("Drawing a plot")
    for iteration in range(iterations):
        if iteration == progress[1]:
            print("%s%%" % progress[0], end=" ")
            progress[0] += 5
            progress[1] += iterations // 20
        model.weights, model.diff_weights, converged = grad_step(model, iteration, grad_function)
        if iteration > 5:
            error = smape(dataset, model.weights)
            losses.append((iteration, error))
        if converged:
            break
    iterations = list(map(lambda x: x[0], losses))
    errors = list(map(lambda y: y[1], losses))
    plt.title("%s, regularisation = %s" % (grad_type, regularisation))
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.plot(iterations, errors)
    plt.show()
    print("\n")
