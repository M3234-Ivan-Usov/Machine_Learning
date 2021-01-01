from math import sqrt
import numpy as np


def euclidean(sample1, sample2):
    return sqrt(np.sum([(sample1[x] - sample2[x]) ** 2 for x in range(len(sample1))]))


def manhattan(sample1, sample2):
    return np.sum([abs(sample1[x] - sample2[x]) for x in range(len(sample1))])


def jaccard_index(clusters, dataset, cluster_amount):
    tp, fp, fn = 0, 0, 0
    for x in range(dataset.objects_amount):
        for y in range(x):
            if clusters[x] == 0 or clusters[y] == 0:  # noise
                fp += 1
            elif clusters[x] == clusters[y] and dataset.targets[x] == dataset.targets[y]:
                tp += 1
            elif clusters[x] == clusters[y] and dataset.targets[x] != dataset.targets[y]:
                fp += 1
            elif clusters[x] != clusters[y] and dataset.targets[x] == dataset.targets[y]:
                fn += 1
    return tp / (tp + fp + fn)


def f_measure(clusters, dataset, clusters_amount):
    actual_amount, atom = len(dataset.unique), 1 / dataset.objects_amount
    conjugacy_matrix = np.zeros(shape=(actual_amount, clusters_amount))
    for sample in range(dataset.objects_amount):
        actual = dataset.vectorised[dataset.targets[sample]]
        conjugacy_matrix[actual, clusters[sample]] += atom
    cluster_sums = [np.sum(conjugacy_matrix[:, y]) for y in range(clusters_amount)]
    actual_sums = [np.sum(conjugacy_matrix[x, :]) for x in range(actual_amount)]
    f_score = 0.0
    for cluster_index in range(1, clusters_amount):
        max_score = 0.0
        for actual_index in range(actual_amount):
            pre_div, re_div = actual_sums[actual_index], cluster_sums[cluster_index]
            precision = conjugacy_matrix[actual_index, cluster_index] / pre_div if pre_div != 0.0 else 0.0
            recall = conjugacy_matrix[actual_index, cluster_index] / re_div if re_div != 0.0 else 0.0
            if precision + recall != 0.0:
                local_score = (2 * precision * recall) / (precision + recall)
                max_score = max(max_score, local_score)
        f_score += cluster_sums[cluster_index] * max_score
    return f_score


def to_cluster(sample, cluster, distances):
    return np.sum([y[0] for y in filter(lambda x: x[1] in cluster, distances[sample])]) / len(cluster)


def separable(sample, expelled, clusters, distances):
    return min([to_cluster(sample, cluster, distances) for cluster in filter(lambda cl: cl != expelled, clusters)])


def silhouette(clusters, distances):
    score = 0.0
    for cluster in clusters:
        for sample in cluster:
            compactness = np.sum([x[0] for x in filter(lambda y: y[1] in cluster, distances[sample])])
            separability = separable(sample, cluster, clusters, distances)
            divisor = max(separability, compactness)
            score += (separability - compactness) / divisor if divisor != 0.0 else 0.0
    return score / len(distances)


def cluster_diameter(cluster, distances):
    if len(cluster) == 1:
        return 0.0
    maximals = [next(filter(lambda x: x[1] in cluster, reversed(distances[sample]))) for sample in cluster]
    return max(maximals, key=lambda x: x[0])[0]


def min_cluster_dist(cluster, distances):
    minimals = [next(filter(lambda x: x[1] not in cluster, distances[sample])) for sample in cluster]
    return min(minimals, key=lambda x: x[0])[0]


def dunn_index(clusters, distances):
    diameters = [cluster_diameter(cluster, distances) for cluster in clusters]
    dist_to_other = [min_cluster_dist(cluster, distances) for cluster in clusters]
    return min(dist_to_other) / max(diameters)


outer_measure = f_measure
inner_measure = dunn_index
