import numpy as np

import estimation
import functions
import gradient
import least_squares
import structures


def parse(offset, objects_amount):
    features = np.empty(shape=(objects_amount, features_amount), dtype=float)
    targets = np.empty(shape=objects_amount)
    for i in range(objects_amount):
        row = data[i + offset].split()
        for j in range(features_amount):
            features[i, j] = int(row[j])
        targets[i] = row[features_amount]
    return features, targets


def add_const_feature(features):
    temp = list()
    temp.append(np.ones(shape=len(features), dtype=float))
    for col in range(len(features[0])):
        temp.append(features[:, col])
    return np.array(temp).transpose()


def seek_identical(features):
    useless_features = np.zeros(shape=features_amount, dtype=bool)
    for col in range(features_amount):
        if len(np.unique(features[:, col])) == 1:
            useless_features[col] = True
    return useless_features


def delete_identical(features):
    temp = list()
    for col in range(features_amount):
        if useless_features[col]:
            continue
        else:
            temp.append(features[:, col])
    new_dataset = np.array(temp).transpose()
    return new_dataset


def return_useless(approximation):
    new_approximation = np.empty(shape=features_amount + 1)
    j = 0
    new_approximation[0] = approximation[0][0]
    for i in range(features_amount):
        if not useless_features[i]:
            new_approximation[i + 1] = approximation[0][j + 1]
            j += 1
        else:
            new_approximation[i + 1] = 0.0
    return new_approximation, approximation[1], approximation[2]


def add_features(approximations):
    return map(lambda x: (return_useless(x[0]), x[1]), approximations)


file = open("LR/1.txt", "r")
data = file.readlines()
file.close()
features_amount = int(data[0])
train_amount = int(data[1])

train_features, train_targets = parse(2, train_amount)
useless_features = seek_identical(train_features)
train_features = delete_identical(train_features)
train_features = add_const_feature(train_features)
train_dataset = structures.Dataset(train_features, train_targets)

best_squares = return_useless(least_squares.compute(train_dataset, functions.ridge_function))

stochastic_approximations = list()
classic_approximations = list()
mini_batch_approximations = list()

regularisation = estimation.initialise_regularisation()
while estimation.has_next(regularisation):
    stochastic_approximations.append(return_useless(gradient.compute(train_dataset, regularisation, "stochastic")))
    classic_approximations.append(return_useless(gradient.compute(train_dataset, regularisation, "classic")))
    mini_batch_approximations.append(return_useless(gradient.compute(train_dataset, regularisation, "mini_batch")))
    regularisation = estimation.next_coefficient(regularisation)

stochastic_approximations.sort(key=lambda x: x[2])
classic_approximations.sort(key=lambda x: x[2])
mini_batch_approximations.sort(key=lambda x: x[2])
best_stochastic = stochastic_approximations[0][0], stochastic_approximations[0][1]
best_classic = classic_approximations[0][0], classic_approximations[0][1]
best_mini_batch = mini_batch_approximations[0][0], mini_batch_approximations[0][1]

test_amount = int(data[train_amount + 2])
test_features, test_targets = parse(train_amount + 3, test_amount)
test_features = add_const_feature(test_features)
test_dataset = structures.Dataset(test_features, test_targets)

print("\n\n=== Estimating least squares ===")
estimation.estimate(test_dataset, best_squares)
print("=== Estimating stochastic gradient ===")
estimation.estimate(test_dataset, best_stochastic)
print("=== Estimating classic gradient ===")
estimation.estimate(test_dataset, best_classic)
print("=== Estimating mini-batch gradient ===")
estimation.estimate(test_dataset, best_mini_batch)

estimation.draw_plot(test_dataset, best_stochastic[1], "stochastic")
estimation.draw_plot(test_dataset, best_classic[1], "classic")
estimation.draw_plot(test_dataset, best_mini_batch[1], "mini_batch")
estimation.draw_plot(train_dataset, be)
