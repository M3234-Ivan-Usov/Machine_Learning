import numpy as np
import pandas as pd
import pylab
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

ROOT = "DT_csv/"
DATASET_FIRST = 2
DATASET_LAST = 2

MIN_DEPTH = 2
MAX_DEPTH = 10

MIN_BRANCHES = 2
MAX_BRANCHES = 7


class Dataset:
    def __init__(self, dataset, targets_id=None):
        self.obj_amount = len(dataset)
        self.features_amount = len(dataset[0]) - 1
        self.features = dataset[:, :-1]
        self.targets = dataset[:, self.features_amount]
        self.__map_targets(targets_id)

    def __map_targets(self, targets_id):
        if targets_id is None:
            self.new_targets, target_id = dict(), 0
            for target in np.unique(self.targets):
                self.new_targets[target] = target_id
                target_id += 1
        else:
            self.new_targets = targets_id
        self.targets = np.fromiter(map(lambda x: self.new_targets[x], self.targets), dtype=int)


def make_dataset(number):
    if len(number) == 1:
        number = '0' + number
    train_file = open(ROOT + number + "_train.csv")
    test_file = open(ROOT + number + "_test.csv")
    train_dataset = pd.read_csv(train_file).values
    test_dataset = pd.read_csv(test_file).values
    train = Dataset(train_dataset)
    test = Dataset(test_dataset, train.new_targets)
    return train, test


def estimate(actual, expected):
    amount = len(expected)
    ok = np.sum(np.fromiter(map(lambda obj: actual[obj] == expected[obj], range(amount)), dtype=int))
    return ok / amount


def analyse(tree_results):
    analyse_tree(min(tree_results, key=lambda x: x[1][1]))
    analyse_tree(max(tree_results, key=lambda x: x[1][1]))


def analyse_tree(optimal):
    dataset, branches = optimal[0], optimal[1][2]
    cached_result = optimal[2]
    quality_on_depth = list(filter(lambda x: x[2] == branches, cached_result))
    quality_on_depth.sort(key=lambda x: x[1])
    quality = list(map(lambda x: x[0] * 100, quality_on_depth))
    depth = list(map(lambda x: x[1], quality_on_depth))
    draw_plot(quality, depth, dataset)
    draw_surface(cached_result)


def draw_plot(quality, depth, dataset):
    plt.title("Depth impact on accuracy. Dataset #%s" % dataset)
    plt.grid("True")
    plt.xlabel("Tree Max Depth")
    plt.ylabel("Quality, %")
    plt.plot(depth, quality, 'r')
    plt.show()


def draw_surface(data):
    x = np.arange(MIN_DEPTH, MAX_DEPTH + 1, 1.0)
    y = np.arange(MIN_BRANCHES, MAX_BRANCHES + 1)
    x_grid, y_grid = np.meshgrid(x, y)
    x_size = MAX_DEPTH - MIN_DEPTH + 1
    y_size = MAX_BRANCHES - MIN_BRANCHES + 1
    z_grid = x_grid + y_grid  # just to define z_grid type
    for depth in range(x_size):
        x_slice = list(filter(lambda pair: pair[1] == x[depth], data))
        for branch in range(y_size):
            val = list(filter(lambda pair: pair[2] == y[branch], x_slice))
            z_grid[branch][depth] = val[0][0]
    fig = pylab.figure()
    axes = Axes3D(fig)
    axes.plot_surface(x_grid, y_grid, z_grid, cmap=cm.get_cmap(name="Spectral"))
    pylab.show()
