import multiprocessing

from DT.data_manager import MAX_BRANCHES
from DT.data_manager import MAX_DEPTH
from DT.data_manager import MIN_BRANCHES
from DT.data_manager import MIN_DEPTH
from DT.data_manager import estimate
from DT.data_manager import make_dataset
from DT.decision_tree import DecisionTree


class TreeManager(multiprocessing.Process):
    def __init__(self, tasks_queue, results_queue):
        super().__init__()
        self.tasks = tasks_queue
        self.results = results_queue

    def run(self) -> None:
        while not self.tasks.empty():
            self.results.put(self.tasks.get().run())
            self.tasks.task_done()


class TreeTask:
    def __init__(self, dataset):
        self.dataset = dataset

    def run(self):
        print("===== Start processing dataset #%s =====" % self.dataset)
        train, test = make_dataset(str(self.dataset))
        local_results = list()
        for max_depth in range(MIN_DEPTH, MAX_DEPTH + 1):
            process = (max_depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
            print("----- %s%% on dataset #%s -----" % (process * 100, self.dataset))
            for branches in range(MIN_BRANCHES, MAX_BRANCHES + 1):
                tree = DecisionTree(train, max_depth, branches)
                predictions = tree.predict(test.features)
                quality = estimate(predictions, test.targets)
                local_results.append((quality, max_depth, branches))
        the_best_local = max(local_results, key=lambda x: x[0])
        print("===== Finished dataset #%s =====" % self.dataset)
        return self.dataset, the_best_local, local_results
