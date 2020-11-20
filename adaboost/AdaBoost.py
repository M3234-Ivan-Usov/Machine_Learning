import copy
import os
from math import exp, log
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
import numpy as np

RESOLUTION = 150
TREE_DEPTH = 1


class AdaBoost:
    def __init__(self, algo_amount, classifier):
        self.algo_amount = algo_amount
        self.classifier = classifier
        self.checkpoints = self.__fibonacci()
        self.base_algorithms = list()

    def __fibonacci(self):
        f_2, f_1 = 0, 1
        f = set()
        while f_2 <= self.algo_amount:
            f_0 = f_1 + f_2
            f.add(f_0)
            f_2 = f_1
            f_1 = f_0
        return f

    def __update(self, weights, alpha, expected, actual):
        for obj in range(len(expected)):
            weights[obj] *= exp(-alpha * expected[obj] * actual[obj])
        normalised_factor = np.sum(weights)
        return weights / normalised_factor

    def __estimate_quality(self, dataset):
        predictions = self.predict(dataset.features)
        ok = np.sum([predictions[x] == dataset.targets[x] for x in range(dataset.amount)])
        return ok / dataset.amount

    def predict(self, features):
        samples = len(features)
        weighted_predictions = np.zeros(samples)
        for algo in self.base_algorithms:
            weighted_predictions += algo[0] * algo[1].predict(features)
        return np.fromiter(map(np.sign, weighted_predictions), dtype=int)

    def boost(self, dataset, animation_root):
        if animation_root is not None:
            animator = AnimatedScatter(dataset, RESOLUTION, animation_root)
        sample_weights = np.repeat(1 / dataset.amount, dataset.amount)
        quality = list()
        for iteration in range(self.algo_amount + 1):
            classifier = copy.deepcopy(self.classifier)
            classifier.fit(dataset.features, dataset.targets, sample_weights)
            predictions = classifier.predict(dataset.features)
            loss = np.sum([(predictions[x] != dataset.targets[x]) * sample_weights[x]
                           for x in range(dataset.amount)])
            alpha = 0.5 * log((1 - loss) / loss, exp(1))
            sample_weights = self.__update(sample_weights, alpha, dataset.targets, predictions)
            self.base_algorithms.append((alpha, classifier))
            if (iteration + 1) in self.checkpoints:
                quality.append((iteration + 1, self.__estimate_quality(dataset)))
                if animation_root is not None:
                    animator.draw_divisor(self, iteration + 1)
                print("Pass %s iteration" % (iteration + 1))
        return quality


class AnimatedScatter:
    def __init__(self, dataset, resolution, animation_root):
        self.animation_root = animation_root
        if os.path.exists(animation_root):
            for file in os.listdir(animation_root):
                os.remove(animation_root + '/' + file)
        else:
            os.mkdir(animation_root)
        x_min, y_min = np.amin(dataset.features, 0)
        x_max, y_max = np.amax(dataset.features, 0)
        x_step, y_step = (x_max - x_min) / RESOLUTION, (y_max - y_min) / resolution
        self.x_min, self.x_max = x_min - 3 * x_step, x_max + 3 * x_step
        self.y_min, self.y_max = y_min - 3 * y_step, y_max + 3 * y_step
        self.x_cell, self.y_cell = np.meshgrid(
            np.arange(self.x_min, self.x_max, x_step),
            np.arange(self.y_min, self.y_max, y_step))
        self.sample_color = ["g" if target == 1 else "b" for target in dataset.targets]
        self.x_values = dataset.features[:, 0]
        self.y_values = dataset.features[:, 1]
        self.scatter_counter = 100

    def draw_divisor(self, ada, iteration):
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        plt.title("Classification after %s iteration" % iteration)
        mesh_dots = np.c_[self.x_cell.ravel(), self.y_cell.ravel()]
        c = np.apply_along_axis(lambda x: ada.predict([x])[0], 1, mesh_dots)
        c = np.array(c).reshape(self.x_cell.shape)
        plt.pcolormesh(self.x_cell, self.y_cell, c, cmap=ListedColormap(['#9090FF', '#90FF90']), shading='nearest')
        plt.scatter(self.x_values, self.y_values, c=self.sample_color)
        self.scatter_counter += 1
        plt.savefig(self.animation_root + '/' + str(self.scatter_counter))
