import multiprocessing
from DT.tree_manager import TreeManager
from DT.tree_manager import TreeTask
from DT.data_manager import DATASET_FIRST
from DT.data_manager import DATASET_LAST
from DT.data_manager import analyse
from DT.tree_manager import TreeTask

CORES = multiprocessing.cpu_count()
MULTI_PROCESS = True

#  Only a single tree
if __name__ == "__main__" and MULTI_PROCESS:
    print("===== Forking %s processes =====" % CORES)
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    workers = [TreeManager(tasks, results) for core in range(CORES)]
    for worker in workers:
        worker.start()
    for dataset in range(DATASET_FIRST, DATASET_LAST + 1):
        tasks.put(TreeTask(dataset))
    tasks.join()
    data_amount = DATASET_LAST - DATASET_FIRST + 1
    global_results = [results.get() for i in range(data_amount)]
    analyse(global_results)

if not MULTI_PROCESS:
    global_results = [TreeTask(dataset).run() for dataset in range(DATASET_FIRST, DATASET_LAST + 1)]
    analyse(global_results)
