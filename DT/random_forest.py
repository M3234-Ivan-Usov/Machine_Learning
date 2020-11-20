import random
from math import sqrt
from matplotlib import pyplot as plt
from DT.data_manager import Dataset
from DT.data_manager import estimate
from DT.data_manager import make_dataset
from DT.decision_tree import DecisionTree
import numpy as np

TREES_IN_FOREST = 50
MAX_DEPTH = 100
BINARY_TREE = 2
DATASET = 2


def make_tree(general, tree):
    features_sub_amount = int(sqrt(general.features_amount))
    selected_samples = [random.randint(0, general.obj_amount - 1) for _ in range(general.obj_amount)]
    selected_features = set()
    while len(selected_features) != features_sub_amount:
        selected_features.add(random.randint(0, general.features_amount - 1))
    features = np.take(general.features, selected_samples, axis=0)
    targets = np.take(general.targets, selected_samples)
    cut_features = np.take(features, list(selected_features), axis=1)
    tree_dataset = Dataset(np.concatenate((cut_features, targets[:, None]), axis=1))
    print("Building tree #%s" % tree)
    return DecisionTree(tree_dataset, MAX_DEPTH, BINARY_TREE), selected_features


def vote(single_predictions):
    amount = len(single_predictions[0])
    cooperative_predictions = np.empty(amount, dtype=int)
    for obj in range(amount):
        targets, votes = np.unique(single_predictions[:, obj], return_counts=True)
        major = targets[np.argmax(votes)]
        cooperative_predictions[obj] = major
    return cooperative_predictions


def draw_plot(test, train):
    x = np.arange(1, TREES_IN_FOREST + 1)
    plt.title("Influence of forest size on accuracy")
    plt.xlabel("Trees in forest")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.plot(x, train, 'b', label="train")
    plt.plot(x, test, 'r', label="test")
    plt.legend()
    plt.show()


test_quality, train_quality = list(), list()
general_train, general_test = make_dataset(str(DATASET))
forest = list()
for forest_size in range(1, TREES_IN_FOREST + 1):
    forest.append(make_tree(general_train, forest_size))
    test_predictions = np.empty((forest_size, general_test.obj_amount))
    train_predictions = np.empty((forest_size, general_test.obj_amount))
    for tree in range(forest_size):
        test_features = np.take(general_test.features, list(forest[tree][1]), axis=1)
        train_features = np.take(general_train.features, list(forest[tree][1]), axis=1)
        test_predictions[tree] = forest[tree][0].predict(test_features)
        train_predictions[tree] = forest[tree][0].predict(train_features)
    test_quality.append(estimate(vote(test_predictions), general_test.targets))
    train_quality.append(estimate(vote(train_predictions), general_train.targets))
draw_plot(test_quality, train_quality)
