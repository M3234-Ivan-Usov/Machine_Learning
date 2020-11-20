from _collections import defaultdict
from math import log

import numpy as np

CLASSES_AMOUNT = 2


class BayesClassifier:
    def __init__(self, dataset, params):
        self.dataset = dataset
        self.lambdas = [1.0, 1.0 * params.lambda_ratio]
        self.alpha = params.alpha
        self.subject_weight = params.subject_weight
        self.n_gramm = params.n_gramm
        self.__calc_frequencies()

    def __calc_frequencies(self):
        self.frequencies = [defaultdict(float) for i in range(CLASSES_AMOUNT)]
        self.subject_frequencies = [defaultdict(float) for i in range(CLASSES_AMOUNT)]
        words_amount, self.targets = 0, np.zeros(CLASSES_AMOUNT, dtype=float)
        for letter in self.dataset:
            #  words, counts = np.unique(letter[2], return_counts=True)
            letter_size, letter_type = len(letter[2]), letter[0]
            words_amount += letter_size
            self.targets[letter_type] += 1
            for word in letter[2]:
                self.frequencies[letter_type][word] += 1
            for word in letter[1]:
                self.subject_frequencies[letter_type][word] += 1
        objects_amount = len(self.dataset)
        for letter_type in range(CLASSES_AMOUNT):
            for word in self.frequencies[letter_type]:
                self.frequencies[letter_type][word] /= words_amount
            self.targets[letter_type] /= objects_amount
        self.alpha /= words_amount

    def classify(self, tests):
        predictions = list()
        for test in tests:
            this_letter = list()
            for predicting_type in range(CLASSES_AMOUNT):
                classifier = log(self.lambdas[predicting_type] * self.targets[predicting_type])
                for word in test[1]:
                    classifier += log(self.frequencies[predicting_type][word] + self.alpha)
                for word in test[0]:
                    classifier += self.subject_weight * log(
                        self.subject_frequencies[predicting_type][word] + self.alpha)
                this_letter.append((predicting_type, classifier))
            this_letter.sort(key=lambda x: x[1], reverse=True)
            predictions.append(this_letter[0][0])
        return predictions

    def probabilities(self, tests):
        probabilities = list()
        NEGATIVE, POSITIVE = 0, 1
        for test in tests:
            positive_classifier = log(self.lambdas[POSITIVE] * self.targets[POSITIVE])
            negative_classifier = log(self.lambdas[NEGATIVE] * self.targets[NEGATIVE])
            for word in test[1]:
                positive_classifier += log(self.frequencies[POSITIVE][word] + self.alpha)
                negative_classifier += log(self.frequencies[NEGATIVE][word] + self.alpha)
            for word in test[0]:
                positive_classifier += self.subject_weight * log(
                    self.subject_frequencies[POSITIVE][word] + self.alpha)
                negative_classifier += self.subject_weight * log(
                    self.subject_frequencies[NEGATIVE][word] + self.alpha)
            if positive_classifier > 0 and negative_classifier > 0:
                probabilities.append(positive_classifier / negative_classifier)
            else:
                probabilities.append(negative_classifier / positive_classifier)
        return probabilities
