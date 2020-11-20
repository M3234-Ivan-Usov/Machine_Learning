import os
from math import log

import numpy as np
from bayes.bayes_classifier import BayesClassifier
from bayes.bayes_classifier import CLASSES_AMOUNT
import matplotlib.pyplot as plt

SUBJECT_PREFIX = 9
ROOT = "messages"
PARTS = 10

ALPHAS_MAX_EXP = -5
ALPHAS_MIN_EXP = -5
LAMBDA_MIN = 0
LAMBDA_MAX = 250
LAMBDA_STEP = 25
N_GRAMM_MAX = 1

ALPHAS = [10 ** x for x in range(ALPHAS_MIN_EXP, ALPHAS_MAX_EXP + 1)]
LAMBDAS_RATIO = [10 ** x for x in range(LAMBDA_MIN, LAMBDA_MAX + 1, LAMBDA_STEP)]
SUBJECT_RATIO = [0.5, 1.0, 2.0]

ROC_RESOLUTION = 100


def make_dataset(root, without, n_gramm):
    train_dataset, test_contents, test_targets = list(), list(), list()
    for part in os.listdir(root):
        current_dir = root + '/' + part
        for file in os.listdir(current_dir):
            source = open(current_dir + '/' + file)
            subject = np.fromiter(source.readline()[SUBJECT_PREFIX:].split(), dtype=int)
            source.readline()
            content = split_n_gramms(np.fromiter(source.readline().split(), dtype=int), n_gramm)
            letter_type = 1 if "legit" in file else 0
            if part == without:
                test_contents.append((subject, content))
                test_targets.append(letter_type)
            else:
                train_dataset.append((letter_type, subject, content))
            source.close()
    return train_dataset, test_contents, test_targets


def split_n_gramms(raw_content, n_gramm):
    new_size = len(raw_content) - n_gramm
    content = list()
    for i in range(new_size):
        bunch = [raw_content[i + j] for j in range(n_gramm)]
        content.append(tuple(bunch))
    return content


def f_measure(expected, actual):
    amount = len(expected)
    confusion_matrix = np.zeros((CLASSES_AMOUNT, CLASSES_AMOUNT), dtype=int)
    for letter in range(amount):
        confusion_matrix[actual[letter], expected[letter]] += 1
    tp, fp, fn = 0, 0, 0
    for x in range(CLASSES_AMOUNT):
        tp += confusion_matrix[x, x]
        fp += np.sum(confusion_matrix[x, :]) - confusion_matrix[x, x]
        fn += np.sum(confusion_matrix[:, x]) - confusion_matrix[x, x]
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall), confusion_matrix[0, 1] / amount


def draw_plot(results):
    extracted = list()
    for lambda_ in LAMBDAS_RATIO:
        for result in results:
            if result.lambda_ratio == lambda_:
                extracted.append((result.lambda_ratio, result.quality_average, result.selection_average * 100))
                break
    extracted.sort(key=lambda x: x[0])
    keys = list(map(lambda x: x[0], extracted))
    f_measures = list(map(lambda x: x[1], extracted))
    fails = list(map(lambda x: x[2], extracted))

    fig, ax1 = plt.subplots(1, 1)
    plt.xscale("log")
    plt.grid(True)
    ax1.plot(keys, f_measures, color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(keys, fails, color="tab:blue")
    ax1.set_xlabel("Lambda for legit")
    ax1.tick_params(axis="x")
    ax1.set_ylabel("F-measure", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylabel("Legits classified as spam, %", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    plt.show()


def roc_analysis(params):
    probabilities, POSITIVE = list(), 1
    for part in range(1, PARTS + 1):
        train, test, targets = make_dataset(ROOT, "part" + str(part), params.n_gramm)
        classifier = BayesClassifier(train, params)
        probabilities.extend(zip(classifier.probabilities(test), targets))
    min_param = min(probabilities, key=lambda x: x[0])
    max_param = max(probabilities, key=lambda x: x[0])
    threshold, roc_step = min_param[0], (max_param[0] - min_param[0]) / ROC_RESOLUTION
    positives = len(list(filter(lambda x: x[1] == POSITIVE, probabilities)))
    tpr, fpr = list(), list()
    total = len(probabilities)
    negatives = total - positives
    while threshold <= max_param[0]:
        above_threshold = list(filter(lambda x: x[0] >= threshold, probabilities))
        tp = len(list(filter(lambda x: x[1] == POSITIVE, above_threshold)))
        tpr.append(tp / positives)
        fpr.append((len(above_threshold) - tp) / negatives)
        threshold += roc_step

    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.title("ROC-Curve for lambda power = %s" % log(params.lambda_ratio, 10))
    plt.grid(True)
    plt.plot(fpr, tpr, 'r')
    plt.show()


class HyperParams:
    def __init__(self, alpha, lambda_ratio, subject_weight, n_gramm):
        self.alpha = alpha
        self.lambda_ratio = lambda_ratio
        self.subject_weight = subject_weight
        self.n_gramm = n_gramm
        self.quality_average = 0.0
        self.selection_average = 0.0

    def put_result(self, quality_average, selection_average):
        self.quality_average = quality_average
        self.selection_average = selection_average

    def __str__(self):
        return "\n--- F-measure: " + str(self.quality_average) + \
               "\n--- Legits classified as spam: " + str((self.selection_average * 100)) + \
               "\n--- Alpha: " + str(self.alpha) + \
               "\n--- Lambda: " + str(self.lambda_ratio) + \
               "\n--- Subject weight: " + str(self.subject_weight) + \
               "\n--- N-gramm: " + str(self.n_gramm)


global_quality = list()
print("===== Starting calculations =====")
for lambda_ratio in LAMBDAS_RATIO:
    for alpha in ALPHAS:
        for subject_weight in SUBJECT_RATIO:
            for n_gramm in range(1, N_GRAMM_MAX + 1):
                params = HyperParams(alpha, lambda_ratio, subject_weight, n_gramm)
                print("Alpha = %s, Lambda Ratio = %s, Subject Weight = %s, N-gramm = %s" % (
                    alpha, lambda_ratio, subject_weight, n_gramm))
                lambdas = [1.0, 1.0 * lambda_ratio]
                local_quality = list()
                for part in range(1, PARTS + 1):
                    train, test, targets = make_dataset(ROOT, "part" + str(part), n_gramm)
                    predictions = BayesClassifier(train, params).classify(test)
                    quality, legit_to_spam = f_measure(targets, predictions)
                    local_quality.append((quality, legit_to_spam))
                quality_average = np.average(np.fromiter(map(lambda x: x[0], local_quality), dtype=float))
                selection_average = np.average(np.fromiter(map(lambda x: x[1], local_quality), dtype=float))
                params.put_result(quality_average, selection_average)
                global_quality.append(params)

print("\n===== Finished. Processing results =====")
best_legit = min(global_quality, key=lambda x: x.selection_average)
print("\n=== The best selection===%s" % best_legit)

global_quality.sort(key=lambda x: x.quality_average, reverse=True)
best_quality = global_quality[0]
print("\n=== The best quality===%s" % best_quality)

print("\n===== Drawing plots =====")
roc_analysis(best_quality)
draw_plot(global_quality)
