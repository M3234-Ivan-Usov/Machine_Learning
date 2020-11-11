from math import sqrt

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import params
import time


def numerate_classes():
    for row in dataset:
        row[class_index] = classes_map[row[class_index]]


def minmax(dataset):
    minmax = list()
    features_amount = len(dataset[0])
    for i in range(features_amount):
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max()
        minmax.append([value_min, value_max])
    return minmax


def normalize(dataset, minmax):
    normalized_dataset = np.array(dataset)
    for row in normalized_dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return normalized_dataset


def normalize_input(target, minmax):
    normalized_target = np.array(target)
    for i in range(len(target)):
        normalized_target[i] = (normalized_target[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return normalized_target


def kernel_function(current_neighbour, last_neighbour, ker_type, fix_win=False, win_size=1.0):
    numerator = current_neighbour[1]
    denominator = win_size if fix_win else last_neighbour[1]
    return params.kernel(ker_type)(numerator / denominator)


def nadaraya_watson(row, neighbours, ker_type, dist_type, k_value, fixed_window=False, window_size=1.0):
    weighted_sum = np.zeros(classes_amount, dtype=float)
    kernel_sum = 0.0
    environment = neighbours[row, dist_type]
    kth_neighbour = environment[k_value]
    for neighbour in environment[:k_value]:
        weight = kernel_function(neighbour, kth_neighbour, ker_type, fix_win=fixed_window, win_size=window_size)
        current_class = int(neighbour[0]) - 1
        weighted_sum[current_class] += weight
        kernel_sum += weight
    weighted_sum /= kernel_sum
    return weighted_sum


def f_measure(confusion_matrix, all):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(classes_amount):
        for j in range(classes_amount):
            if i == j:
                tp += confusion_matrix[i, j]
                continue
            if i > j:
                fn += confusion_matrix[i, j]
                continue
            if i < j:
                fp += confusion_matrix[i, j]
                continue
    tn = all - tp - fp - fn
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (F_BETA * precision + recall)


def knn(general_neighbours, ker_type, dist_type, k_value, fixed_window=False, window_size=1.0):
    progress = [objects_amount // 10, objects_amount // 10, 10]
    confusion_matrix = np.zeros((classes_amount, classes_amount), dtype=int)
    for row in range(objects_amount):
        if row == progress[0]:
            print('%s%%' % progress[2], end=" ")
            progress[2] += 10
            progress[0] += progress[1]
        prediction = nadaraya_watson(row, general_neighbours, ker_type, dist_type,
                                     k_value, fixed_window=fixed_window, window_size=window_size)
        prediction_class = int(prediction.argmax()) + 1
        test_class = dataset[row, class_index]
        confusion_matrix[prediction_class - 1, test_class - 1] += 1
    return f_measure(confusion_matrix, objects_amount)


def get_all_neighbours(dataset):
    neighbours = np.zeros((objects_amount, DISTANCES), dtype=list)
    progress = [objects_amount // 10, objects_amount // 10, 10]
    maximal_distance = 0.0
    for row1 in range(objects_amount):
        if row1 == progress[0]:
            print('%s%%' % progress[2], end=" ")
            progress[2] += 10
            progress[0] += progress[1]
        leave_one_out_dataset = np.delete(dataset, row1, axis=0)
        current_object = dataset[row1, :-1]
        min_max = minmax(leave_one_out_dataset[:, :-1])
        normalized_dataset = normalize(leave_one_out_dataset, min_max)
        normalized_target = normalize_input(current_object, min_max)
        current_distances = list()
        for dist_type in range(DISTANCES):
            for row2 in normalized_dataset:
                current_distance = params.distance(dist_type)(
                    normalized_target, row2[:-1])
                maximal_distance = max(maximal_distance, current_distance)
                current_distances.append((row2[class_index], current_distance))
            current_distances.sort(key=lambda tup: tup[1])
            neighbours[row1, dist_type] = current_distances
    return neighbours, maximal_distance


def draw_plot_neighbours(ker_type, dist_type):
    y = list(efficiency[ker_type, dist_type, 1:, 0])
    x = list(range(1, NEIGHBOURS))
    plt.title('Kernel: %s, distance: %s' % (kernels_map[ker_type], distances_map[dist_type]))
    plt.xlabel("K-Value")
    plt.ylabel("F-measure")
    plt.plot(x, y)
    plt.show()


def draw_plot_windows(ker_type, dist_type, k_value):
    y = list(efficiency[ker_type, dist_type, k_value, 1:])
    x = list()
    window_size = window_step
    while window_size <= WINDOWS:
        x.append(window_size)
        window_size += window_step
    plt.title('Kernel: %s, distance: %s, K-Value: %s' % (kernels_map[ker_type], distances_map[dist_type], k_value))
    plt.xlabel("Window Size")
    plt.ylabel("F-measure")
    plt.plot(x, y)
    plt.show()


filename = 'Vehicle.csv'
F_BETA, KERNELS, DISTANCES = (1, 5, 3)
kernels_map = {0: 'uniform', 1: 'triangular', 2: 'epanenchikov', 3: 'quartic', 4: 'gaussian'}
distances_map = {0: 'euclidean', 1: 'manhattan', 2: 'chebyshev'}

dataset = np.array(pd.read_csv(filename).values)
objects_amount = len(dataset)
class_index = len(dataset[0]) - 1
classes_available = np.unique(dataset[:, class_index])
classes_amount = len(classes_available)
classes_map = dict()
for index in range(classes_amount):
    classes_map[classes_available[index]] = index
numerate_classes()

time_start = time.time()
print("===== Preprocessing =====\nProcess:", end=" ")
general_neighbours, WINDOWS = get_all_neighbours(dataset)
NEIGHBOURS = int(sqrt(objects_amount))
window_step = WINDOWS / sqrt(objects_amount)
efficiency = np.zeros((KERNELS, DISTANCES, NEIGHBOURS, NEIGHBOURS + 1), dtype=float)
print('\n===== Finished in %s s' % (time.time() - time_start))

print('\n===== Start =====')
print('===== Amount of neighbours is %s =====' % (NEIGHBOURS - 1))
print('===== Maximal Window Size is %s =====' % WINDOWS)
print('===== Window Step Size is %s =====\n' % window_step)
time_start = time.time()

for ker in range(KERNELS):
    for dist in range(DISTANCES):
        for k_value in range(1, NEIGHBOURS):
            print('--- Kernel: %s, distance: %s, K-value: %s, window: k-th neighbour ---\nProcess:' %
                  (kernels_map[ker], distances_map[dist], k_value), end=" ")
            efficiency[ker, dist, k_value, 0] = knn(
                general_neighbours, ker, dist, k_value, fixed_window=False)
            print('\nF_measure: %s\n' % efficiency[ker, dist, k_value, 0])

            window_size, position = (window_step, 1)
            while window_size <= WINDOWS:
                print('--- Kernel: %s, distance: %s, K-value: %s, window: %s ---\nProcess:' %
                      (kernels_map[ker], distances_map[dist], k_value, window_size), end=" ")
                efficiency[ker, dist, k_value, position] = knn(
                    general_neighbours, ker, dist, k_value, fixed_window=True, window_size=window_size)
                print('\nF_measure: %s\n' % efficiency[ker, dist, k_value, position])
                position += 1
                window_size += window_step

print('\n===== Finished in %s s =====\n' % (time.time() - time_start))
the_best = np.unravel_index(np.argmax(efficiency, axis=None), efficiency.shape)
print('The best F-measure: %s with' % efficiency[the_best])
the_best_params = np.array(the_best)
print('Kernel type: %s, Distance measure: %s, K-value: %s' %
      (kernels_map[the_best_params[0]], distances_map[the_best_params[1]], the_best_params[2]), end=", ")
if the_best_params[3] == 0:
    print('Window-size: k-th neighbour')
else:
    window_size, position = (window_step, 1)
    while position != the_best_params[3]:
        window_size += window_step
        position += 1
    print('Window-size: %s' % window_size)
draw_plot_neighbours(the_best_params[0], the_best_params[1])
draw_plot_windows(the_best_params[0], the_best_params[1], the_best_params[2])
