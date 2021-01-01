import numpy as np
from functools import reduce
from clust import functions


class Hierarchy:
    class Cluster:
        def __init__(self, samples):
            self.samples = samples
            self.distances = dict()

    def __init__(self, dataset):
        self.dataset = dataset
        self.left_bound = len(dataset.unique) // 2
        self.right_bound = 2 * len(dataset.unique)
        if self.left_bound == 1:
            self.left_bound += 1
            self.right_bound += 1

    def calc_distances(self, clusters):
        initial_distances = [list() for _ in range(self.dataset.objects_amount)]
        total_pairs = self.dataset.objects_amount * (self.dataset.objects_amount - 1)
        progress, step, next_point, proc = 0, total_pairs // 10, total_pairs // 10, 10
        for cluster in clusters:
            for another_cluster in filter(lambda cl: cl != cluster, clusters):
                x, y = min(cluster.samples), min(another_cluster.samples)
                dist = self.dataset.distance_fun(self.dataset.features[x], self.dataset.features[y])
                cluster.distances[another_cluster] = dist
                initial_distances[x].append((dist, y))
                progress += 1
                if progress == next_point:
                    print(" %s%%" % proc, end="")
                    proc += 10
                    next_point += step
        for sample_dist in initial_distances:
            sample_dist.sort(key=lambda z: z[0])
        return initial_distances

    def __call__(self, *args, **kwargs):
        clusters = {Hierarchy.Cluster({sample}) for sample in range(self.dataset.objects_amount)}
        initial_distances = self.calc_distances(clusters)
        measures, best = list(), (0.0, -1.0, None, 0)
        step, next_point, proc = self.dataset.objects_amount // 10, self.dataset.objects_amount // 10, 10
        print("\n===== Processing:", end="")
        for iteration in range(1, self.dataset.objects_amount):
            if iteration == next_point:
                print(" %s%%" % proc, end="")
                next_point += step
                proc += 10
            x, y = self.find_closest(clusters)
            clusters.remove(y)
            clusters.remove(x)
            w = Hierarchy.Cluster(x.samples | y.samples)
            self.update_distances(clusters, x, y, w)
            clusters.add(w)
            if self.left_bound <= len(clusters) <= self.right_bound:
                outer, inner, arrayed = self.estimate(clusters, initial_distances)
                measures.append((outer, inner, len(clusters)))
                if best[0] < outer:
                    best = (outer, inner, arrayed, len(clusters))
        measures.reverse()
        print("\n===== The best outer: %.2f with inner: %.2f for %s clusters =====" % (best[0], best[1], best[3]))
        return best[0], best[1], best[2], measures

    def find_closest(self, clusters):
        minimals = [(cluster, min(cluster.distances.items(), key=lambda x: x[1])) for cluster in clusters]
        absolute_minimal = min(minimals, key=lambda x: x[1][1])
        return absolute_minimal[0], absolute_minimal[1][0]

    def lance_williams(self, s, x, y):
        total = len(s) + len(x) + len(y)
        alpha_x = (len(s) + len(x)) / total
        alpha_y = (len(s) + len(y)) / total
        beta, gamma = -len(s) / total, 0.0
        return alpha_x * self.ward(s, x) + alpha_y * self.ward(s, y) + beta * self.ward(x, y)

    def ward(self, x, y):
        x_centroid = reduce(lambda f, t: f + t, [self.dataset.features[x_sample] for x_sample in x]) / len(x)
        y_centroid = reduce(lambda f, t: f + t, [self.dataset.features[y_sample] for y_sample in y]) / len(y)
        normaliser = (len(x) * len(y)) / (len(x) + len(y))
        return normaliser * self.dataset.distance_fun(x_centroid, y_centroid) ** 2

    def update_distances(self, clusters, x, y, w):
        for cluster in clusters:
            del cluster.distances[x]
            del cluster.distances[y]
            dist = self.lance_williams(cluster.samples, x.samples, y.samples)
            cluster.distances[w] = dist
            w.distances[cluster] = dist

    def estimate(self, clusters, initial_distances):
        listed = [cluster.samples for cluster in clusters]
        arrayed = np.zeros(shape=self.dataset.objects_amount, dtype=int)
        cluster_id = 1
        for cluster in clusters:
            for sample in cluster.samples:
                arrayed[sample] = cluster_id
            cluster_id += 1
        outer = functions.outer_measure(arrayed, self.dataset, cluster_id + 1)
        inner = functions.inner_measure(listed, initial_distances)
        return outer, inner, arrayed
