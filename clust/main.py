from matplotlib import pyplot as plt
import numpy as np
from pandas import read_csv

from clust.dbscan import DBSCAN
from clust.hierarchy import Hierarchy
import clust.functions as functions

filename = 'datasets/vehicle.csv'
distance = functions.manhattan
use_dbscan = True

colours = ["grey",
           "red", "lime", "royalblue",
           "orangered", "springgreen", "slateblue",
           "brown", "lightgreen", "aqua",
           "gold", "turquoise", "navy",
           "sienna", "olive", "darkcyan",
           "indigo", "purple", "magenta"]


class Dataset:
    def __init__(self, csv_values, distance_fun):
        print("===== Preprocessing:", end="")
        self.objects_amount = len(csv_values)
        self.features_amount = len(csv_values[0]) - 1
        self.targets = csv_values[:, self.features_amount]
        self.unique = np.unique(self.targets)
        self.vectorised = {self.unique[x]: x for x in range(len(self.unique))}
        self.features = np.empty((self.objects_amount, self.features_amount), dtype=float)
        self.distance_fun = distance_fun
        for feature in range(self.features_amount):
            minimal = csv_values[:, feature].min()
            maximal = csv_values[:, feature].max()
            for row in range(self.objects_amount):
                self.features[row, feature] = (csv_values[row, feature] - minimal) / (maximal - minimal)


def visualise_cluster(clusters, dataset):
    collapsed, info = pca_collapse(dataset, to_dim=2)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    cluster_colours = [colours[clusters[2][x]] for x in range(dataset.objects_amount)]
    actual_colours = [colours[dataset.vectorised[dataset.targets[x]] + 1] for x in range(dataset.objects_amount)]
    fig.suptitle("%s: %.2f, %s: %.2f" % (
        functions.outer_measure.__name__, clusters[0], functions.inner_measure.__name__, clusters[1]))
    ax1.set_title("Clustered by DBSCAN" if use_dbscan else "Clustered hierarchically")
    ax2.set_title("Actual labels")
    ax1.scatter(collapsed[:, 0], collapsed[:, 1], c=cluster_colours, s=5)
    ax2.scatter(collapsed[:, 0], collapsed[:, 1], c=actual_colours, s=5)
    plt.show()


def visualise_input(dataset, name):
    collapsed, info = pca_collapse(dataset, to_dim=3)
    sample_colours = [colours[dataset.vectorised[x] + 1] for x in dataset.targets]
    ax = plt.axes(projection="3d")
    ax.scatter3D(collapsed[:, 0], collapsed[:, 1], collapsed[:, 2], c=sample_colours, s=5)
    plt.title("%s, representativity: %.2f%%" % (name, info * 100))
    plt.show()


def draw_plot(measures, depend_on):
    measures.sort(key=lambda x: x[2])
    fig, ax1 = plt.subplots(1, 1)
    plt.grid(True)
    ax1.plot([m[2] for m in measures], [m[0] for m in measures], color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot([m[2] for m in measures], [m[1] for m in measures], color="tab:blue")
    ax1.set_xlabel(depend_on)
    ax1.tick_params(axis="x")
    ax1.set_ylabel(functions.outer_measure.__name__, color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylabel(functions.inner_measure.__name__, color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    plt.show()


def pca_collapse(dataset, to_dim):
    values, vectors = np.linalg.eig(dataset.features.transpose().dot(dataset.features))
    marked_values = [(values[x], x) for x in range(dataset.features_amount)]
    marked_values.sort(key=lambda x: abs(x[0]), reverse=True)
    collapsed_transform = np.array([vectors[:, marked_values[x][1]] for x in range(to_dim)]).transpose()
    info = np.sum([marked_values[x][0] for x in range(to_dim)]) / np.sum(values)
    return dataset.features.dot(collapsed_transform), info


dataset = Dataset(np.array(read_csv(open(filename)).values), distance)
visualise_input(dataset, filename)
result = DBSCAN(dataset)() if use_dbscan else Hierarchy(dataset)()
draw_plot(result[3], "Radius" if use_dbscan else "Clusters")
visualise_cluster(result, dataset)
