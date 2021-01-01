from math import sqrt
import numpy as np
from clust import functions

ITERATIONS = 13


class DBSCAN:
    def __init__(self, dataset):
        self.dataset = dataset
        self.min_in_cluster = int(sqrt(self.dataset.objects_amount) / 10) + 1
        self.distances = self.calc_distances()

    def calc_distances(self):
        distances = [list() for _ in range(self.dataset.objects_amount)]  # np.empty(self.dataset.objects_amount)
        total_pairs = (self.dataset.objects_amount * (self.dataset.objects_amount - 1)) / 2
        next_point, step, proc, progress = total_pairs // 10, total_pairs // 10, 10, 0
        for sample in range(self.dataset.objects_amount):
            for other in range(sample):
                dist = self.dataset.distance_fun(self.dataset.features[sample], self.dataset.features[other])
                distances[sample].append((dist, other))
                distances[other].append((dist, sample))
                progress += 1
                if progress == next_point:
                    print(" %s%%" % proc, end="")
                    next_point += step
                    proc += 10
        for sample_dist in distances:
            sample_dist.sort(key=lambda x: x[0])
        return distances

    def __call__(self, *args, **kwargs):
        best_params = (0.0, -1.0, 0.0, np.empty(self.dataset.features_amount), 0)
        radius = self.dataset.features_amount / (40 * self.min_in_cluster)
        measures, step = list(), radius * 2
        print("\n===== Processing:", end="")
        for iteration in range(1, ITERATIONS + 1):
            print(" %.0f%%" % (iteration * 100 / ITERATIONS), end="")
            arrayed, listed = self.scan(radius)
            if len(listed) > 2:
                outer = functions.outer_measure(arrayed, self.dataset, len(listed))
                inner = functions.inner_measure(listed[1:], self.distances)
                measures.append((outer, inner, radius))
                if best_params[0] < outer:
                    best_params = (outer, inner, radius, arrayed, len(listed))
            radius += step

        if len(measures) == 0:
            print("\n===== DBSCAN failed to extract at least two clusters =====")
        else:
            print("\n===== The best outer: %.2f with inner: %.2f" % (best_params[0], best_params[1]))
            print("===== Clusters: %s with min samples involved: %s" % (best_params[4], self.min_in_cluster))
            return best_params[0], best_params[1], best_params[3], measures

    def scan(self, radius):
        processed = np.full(shape=self.dataset.objects_amount, fill_value=False)
        arrayed = np.full(shape=self.dataset.objects_amount, fill_value=-1)
        listed = [set()]  # noise
        current_cluster = 0
        for sample in range(self.dataset.objects_amount):
            if not processed[sample]:
                processed[sample] = True
                neighbours = self.find_neighbours(sample, radius)
                if len(neighbours) < self.min_in_cluster:
                    arrayed[sample] = 0
                    listed[0].add(sample)
                else:
                    current_cluster += 1
                    arrayed[sample] = current_cluster
                    listed.append({sample})
                    self.new_cluster(current_cluster, neighbours, processed, arrayed, listed, radius)
        return arrayed, listed

    def find_neighbours(self, sample, radius):
        nearest, index = list(), 0
        for index in range(self.dataset.objects_amount - 1):
            current_neighbour = self.distances[sample][index]
            if current_neighbour[0] <= radius:
                nearest.append(current_neighbour[1])
            else:
                break
        return nearest

    def new_cluster(self, current_cluster, neighbours, processed, arrayed, listed, radius):
        while True:
            changed = False
            for neighbour in neighbours:
                if not processed[neighbour]:
                    processed[neighbour] = True
                    next_neighbours = self.find_neighbours(neighbour, radius)
                    if len(next_neighbours) >= self.min_in_cluster:
                        neighbours += next_neighbours
                        changed = True
                if arrayed[neighbour] == -1:
                    arrayed[neighbour] = current_cluster
                    listed[current_cluster].add(neighbour)
                if changed:
                    break

            if not changed:
                return
