from math import log
import numpy as np

EPS = 10e-6


class DecisionTree:
    def __init__(self, dataset, max_depth, spans):
        self.dataset = dataset
        self.max_depth = max_depth
        self.spans = spans
        self.unique = len(np.unique(dataset.targets))
        all_objects = set(np.arange(self.dataset.obj_amount))
        self.root = self.__build_node(all_objects, 0)

    class DecisionNode:
        def __init__(self, decision=None, spans=None, major=-1):
            self.major = major
            if major == -1:
                self.next_feature = decision[1]
                self.spans = spans
                first, step = decision[3], decision[4]
                self.feature_split = [first + group * step for group in range(1, spans)]
                self.next_nodes = list()

        def select_branch(self, obj):
            feature = obj[self.next_feature]
            for span in range(self.spans - 1):
                if feature < self.feature_split[span]:
                    return self.next_nodes[span]
            return self.next_nodes[self.spans - 1]

    def __targets_probability(self, selected):
        probabilities = np.zeros(self.unique, dtype=float)
        for index in selected:
            probabilities[self.dataset.targets[index]] += 1.0
        return probabilities / len(selected)

    def __targets_entropy(self, selected):
        probabilities = self.__targets_probability(selected)
        return np.sum(np.fromiter(map(lambda x: 0.0 if x < EPS else -x * log(x, 2), probabilities), dtype=float))

    # Not in use
    def __gini_gain(self, selected):
        probabilities = self.__targets_probability(selected)
        return 1 - np.sum(np.fromiter(map(lambda x: x ** 2, probabilities), dtype=float))

    def __get_major(self, selected):
        probabilities = self.__targets_probability(selected)
        return self.DecisionNode(major=np.argmax(probabilities))

    def __feature_entropy(self, selected, feature):
        feature_extraction = list(map(
            lambda index: (self.dataset.features[index, feature], index), selected))
        feature_extraction.sort(key=lambda x: x[0])
        feature_split, start, step = self.__split_feature(feature_extraction)
        feature_entropy = 0.0
        for span in feature_split:
            span_probability = len(span) / len(selected)
            span_entropy = self.__targets_entropy(list(map(lambda x: x[1], span)))
            feature_entropy += span_probability * span_entropy
        return feature_entropy, feature, feature_split, start, step

    def __split_feature(self, extraction):
        last = len(extraction) - 1
        start, end = extraction[0][0], extraction[last][0]
        step, group = (end - start) / self.spans, 1
        feature_split = [list() for i in range(self.spans)]
        for pair in extraction:
            feature_split[group - 1].append(pair)
            if pair[0] > start + group * step:
                group += 1
        return feature_split, start, step

    def __build_node(self, selected, depth):
        if depth == self.max_depth:
            return self.__get_major(selected)
        features_entropy = [self.__feature_entropy(selected, feature)
                            for feature in range(self.dataset.features_amount)]
        best_info_gain = min(features_entropy, key=lambda x: x[0])
        node = self.DecisionNode(decision=best_info_gain, spans=self.spans)
        for span in range(self.spans):
            new_selected = set(map(lambda x: x[1], best_info_gain[2][span]))
            if len(new_selected) <= self.dataset.obj_amount // 128:
                node.next_nodes.append(self.__get_major(new_selected))
            else:
                node.next_nodes.append(self.__build_node(new_selected, depth + 1))
        return node

    def predict(self, test):
        predictions = list()
        for obj in test:
            current_node = self.root
            while current_node.major == -1:
                current_node = current_node.select_branch(obj)
            predictions.append(current_node.major)
        return predictions
