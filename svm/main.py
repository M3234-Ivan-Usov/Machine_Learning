import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

import svm.kernels as kernels
from matplotlib import pyplot as plt

BETA = [1, 2, 3, 4, 5]
DEGREES = [2, 3, 4, 5]
SVM_CONST = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
ITERATIONS = 500
RESOLUTION = 250
FAIL_MULTIPLICATOR = 3
EPS = 10e-5


class SVM:
    def __init__(self, features, classes, C, kernel, extra_arg=None):
        self.features = features
        self.targets = classes
        self.amount = len(classes)
        self.C = C
        self.kernel = kernel
        self.extra_arg = extra_arg
        self.alpha = np.zeros(self.amount)
        self.b = 0.0
        self.kernel_matrix = self.__init_kernel_matrix()
        self.features_amount = len(features[0])
        self.launch()

    def __init_kernel_matrix(self):
        return np.array([self.kernel(self.features[x], self.features[y], self.extra_arg) for x in range(
            self.amount) for y in range(self.amount)]).reshape((self.amount, self.amount))

    def __svm_function(self, index):
        return np.sum(self.alpha * self.targets * self.kernel_matrix[index])

    def __calc_constraints(self, x, y):
        if classes[x] == classes[y]:
            lowest = max(0, self.alpha[x] + self.alpha[y] - self.C)
            highest = min(self.alpha[x] + self.alpha[y], self.C)
        else:
            lowest = max(0, self.alpha[y] - self.alpha[x])
            highest = min(self.C - self.alpha[x] + self.alpha[y], self.C)
        return lowest, highest

    def __calc_eta(self, x, y):
        cov = self.kernel_matrix[x, y]
        x_self = self.kernel_matrix[x, x]
        y_self = self.kernel_matrix[y, y]
        return 2 * cov - x_self - y_self

    def __condition(self, index, error):
        first = self.targets[index] * error < -EPS and self.alpha[index] < self.C
        second = self.targets[index] * error > EPS and self.alpha[index] > 0
        return first or second

    def __random_not_equal(self, value):
        new_value = np.random.randint(0, self.amount)
        return new_value if new_value != value else self.__random_not_equal(value)

    def __normalise_alpha(self, index, low, high):
        if self.alpha[index] > high:
            self.alpha[index] = high
        if self.alpha[index] < low:
            self.alpha[index] = low

    def __calc_support_object(self, index):
        return np.sum(self.alpha * self.targets * self.kernel_matrix[index]) - self.targets[index]

    def __calc_b(self):
        for index in range(self.amount):
            if EPS < self.alpha[index] < self.C - EPS:
                self.b = self.__calc_support_object(index)
                return
        supports = list()
        for index in range(self.amount):
            if EPS < self.alpha[index]:
                supports.append(self.__calc_support_object(index))
        self.b = np.sum(supports) / len(supports) if len(supports) != 0 else 0.0

    def __init_kernel_row(self, obj):
        return np.fromiter(map(lambda x: self.kernel(
            self.features[x], obj, self.extra_arg), range(self.amount)), dtype=float)

    def launch(self):
        iteration, has_changed = 0, False
        failed = 0
        while iteration < ITERATIONS:
            for i in range(self.amount):
                error_i = self.__svm_function(i) - self.targets[i]
                if self.__condition(i, error_i):
                    j = self.__random_not_equal(i)
                    error_j = self.__svm_function(j) - self.targets[j]
                    low, high = self.__calc_constraints(i, j)
                    eta = self.__calc_eta(i, j)
                    if low == high or eta >= 0:
                        continue
                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                    self.alpha[j] -= self.targets[j] * (error_i - error_j) / eta
                    self.__normalise_alpha(j, low, high)
                    diff_j = alpha_j_old - self.alpha[j]
                    if abs(diff_j) < EPS:
                        continue
                    self.alpha[i] += self.targets[i] * self.targets[j] * diff_j
                    has_changed = True
            iteration += 1
            if has_changed:
                iteration, has_changed = 0, False
                failed += 1
            if failed >= FAIL_MULTIPLICATOR * ITERATIONS:
                break
        self.__calc_b()

    def predict(self, obj):
        classifier = np.sign(np.sum(self.alpha * self.targets * self.__init_kernel_row(obj)) - self.b)
        return classifier if classifier != 0 else np.random.choice([-1, 1])


def validate(svm):
    return np.sum(np.fromiter(map(
        lambda x: svm.predict(features[x]) == svm.targets[x],
        range(svm.amount)), dtype=int)) / svm.amount


def draw_plot(svm, ker_as_str):
    x_min, y_min = np.amin(svm[0].features, 0)
    x_max, y_max = np.amax(svm[0].features, 0)
    x_step, y_step = (x_max - x_min) / RESOLUTION, (y_max - y_min) / RESOLUTION
    x_min, x_max = x_min - 3 * x_step, x_max + 3 * x_step
    y_min, y_max = y_min - 3 * y_step, y_max + 3 * y_step
    x_cell, y_cell = np.meshgrid(np.arange(x_min, x_max, x_step), np.arange(y_min, y_max, y_step))
    mesh_dots = np.c_[x_cell.ravel(), y_cell.ravel()]
    c = np.apply_along_axis(lambda x: svm[0].predict(x), 1, mesh_dots)
    c = np.array(c).reshape(x_cell.shape)
    plt.figure(figsize=(10, 10))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    col = list(map(lambda x: "g" if svm[0].targets[x] == 1 else "b", range(svm[0].amount)))
    plt.pcolormesh(x_cell, y_cell, c, cmap=ListedColormap(['#9090FF', '#90FF90']), shading='nearest')
    plt.scatter(svm[0].features[:, 0], svm[0].features[:, 1], c=col)
    plt.title("Kernel: %s, C = %s, Param = %s, Accuracy = %s%%" %
              (ker_as_str, svm[0].C, svm[0].extra_arg, svm[1] * 100))
    plt.show()


data = open("chips.csv", "r")
dataset = pd.read_csv(data).values
data.close()
features = np.array(dataset[:, :-1])
classes = np.fromiter(map(lambda x: 1 if x == 'P' else -1, dataset[:, len(features[0])]), dtype=int)
linears, polynomials, gaussians = list(), list(), list()
for C in SVM_CONST:
    print("===== Solving for C = %s =====\n----- Linear -----" % C)
    current = SVM(features, classes, C, kernels.linear)
    linears.append((current, validate(current)))
    for degree in DEGREES:
        print("----- Polynomial, degree = %s -----" % degree)
        current = SVM(features, classes, C, kernels.polynomial, extra_arg=degree)
        polynomials.append((current, validate(current)))
    for beta in BETA:
        print("----- Gaussian, beta = %s -----" % beta)
        current = SVM(features, classes, C, kernels.gaussian, extra_arg=beta)
        gaussians.append((current, validate(current)))
linears.sort(key=lambda x: x[1], reverse=True)
polynomials.sort(key=lambda x: x[1], reverse=True)
gaussians.sort(key=lambda x: x[1], reverse=True)
print("===== Drawing plots =====")
draw_plot(linears[0], "Linear")
draw_plot(polynomials[0], "Polynomial")
draw_plot(gaussians[0], "Gaussian")
