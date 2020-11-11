import random
from math import sqrt

import numpy as np

import estimation
import functions
from structures import Model

LAMBDA = 0.01
EPS = 0.1
EPS_DIFF = 0.01

MINI_BATCH_SELECTION = 10

CLASSIC_ITERATIONS = 800
STOCHASTIC_ITERATIONS = 10000
MINI_BATCH_ITERATIONS = 1500


def initialise_weights(dataset):
    weights = np.empty(shape=dataset.features_amount)
    for col in range(dataset.features_amount):
        squared_norm = np.sum(dataset.features[:, col] * dataset.features[:, col])
        weights[col] = np.sum(dataset.targets * dataset.features[:, col]) / squared_norm
    return weights


def configure(grad_type, targets):
    if grad_type == "stochastic":
        print("--- Stochastic Gradient.", end=" ")
        return functions.lasso_function, functions.lasso_grad, stochastic_step, STOCHASTIC_ITERATIONS
    if grad_type == "classic":
        print("--- Classic Gradient.", end=" ")
        return functions.lasso_function, functions.lasso_grad, classic_step, CLASSIC_ITERATIONS
    if grad_type == "mini_batch":
        if len(targets) < 3 * MINI_BATCH_SELECTION:
            print("--- Not enough data for Mini-Batch Gradient\n"
                  "--- Using Stochastic Gradient.", end=" ")
            return functions.lasso_function, functions.lasso_grad, stochastic_step, STOCHASTIC_ITERATIONS
        else:
            print("--- Mini-Batch Gradient.", end=" ")
            return functions.lasso_function, functions.lasso_grad, mini_batch_step, MINI_BATCH_ITERATIONS


def compute(dataset, regularisation, grad_type):
    loss_function, grad_function, grad_step, iterations = configure(grad_type, dataset.targets)
    initial_step = 1 / estimation.get_quartiles(dataset.targets)
    print("Regularisation : %s\nProgress:" % regularisation, end=" ")
    weights = initialise_weights(dataset)
    model = Model(dataset, weights, regularisation, initial_step, EPS + 1.0)
    # error = functions.estimate_loss(features, weights, targets, regularisation, loss_function)
    progress = [5, iterations // 20]
    for iteration in range(iterations):
        if iteration == progress[1]:
            print("%s%%" % progress[0], end=" ")
            progress[0] += 5
            progress[1] += iterations // 20
        model.weights, model.diff_weights, converged = grad_step(model, iteration, grad_function)
        if converged:
            print("\n--- Early Convergence", end="")
            break
    print("\n")
    error = functions.estimate_loss(model, loss_function)
    return model.weights, regularisation, error


def stochastic_step(model, iteration, grad_function):
    gradient_step = model.initial_grad / (iteration + 1)
    selected = random.randint(0, model.data_amount - 1)
    local_gradient = grad_function(model, selected)
    new_weights = model.weights - gradient_step * local_gradient
    diff_weights = functions.l1_norm(new_weights - model.weights)
    if (diff_weights < EPS and model.diff_weights < EPS) or \
            abs(model.diff_weights - diff_weights) < EPS_DIFF:
        return new_weights, diff_weights, True
    return new_weights, diff_weights, False


def classic_step(model, iteration, grad_function):
    gradient_step = model.initial_grad / (iteration + 1)
    local_gradient = np.zeros(shape=model.features_amount, dtype=float)
    for row in range(model.data_amount):
        local_gradient += grad_function(model, row)
    local_gradient /= model.data_amount
    # local_loss = functions.estimate_loss(features, weights, targets, regularisation, loss_function)
    new_weights = model.weights - gradient_step * local_gradient
    # new_error = LAMBDA * local_loss + (1 - LAMBDA) * error
    # new_diff_loss = abs(new_error - error)
    diff_weights = functions.l1_norm(new_weights - model.weights)
    # print("w %s, l %s" % (diff_weights, new_diff_loss))
    if (diff_weights < EPS and model.diff_weights < EPS) or \
            abs(model.diff_weights - diff_weights) < EPS_DIFF:  # and new_diff_loss < EPS_LOSS:
        return new_weights, diff_weights, True
    return new_weights, diff_weights, False


def mini_batch_step(model, iteration, grad_function):
    mini_batch_amount = model.data_amount // MINI_BATCH_SELECTION
    gradient_step = model.initial_grad / (iteration + 1)
    local_gradient = np.zeros(shape=model.features_amount, dtype=float)
    for item in range(mini_batch_amount):
        selected = random.randint(0, model.data_amount - 1)
        local_gradient += grad_function(model, selected)
    local_gradient /= mini_batch_amount
    new_weights = model.weights - gradient_step * local_gradient
    diff_weights = functions.l1_norm(new_weights - model.weights)
    if (diff_weights < EPS and model.diff_weights < EPS) or \
            abs(model.diff_weights - diff_weights) < EPS_DIFF:
        return new_weights, diff_weights, True
    return new_weights, diff_weights, False
