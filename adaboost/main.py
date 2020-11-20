import glob
import pandas as pd

import moviepy.editor as mpy
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from adaboost.AdaBoost import AdaBoost

FILE = "chips"
ANIMATION_ROOT = "animation/"
ITERATIONS = 2800
ANIMATE = False


class Dataset:
    def __init__(self, dataset):
        self.features = dataset[:, :-1]
        self.targets = [1 if x == 'P' else -1 for x in dataset[:, 2]]
        self.amount = len(self.targets)


def draw_plot(quality):
    plt.close()
    steps = [x[0] for x in quality]
    accuracy = [x[1] for x in quality]
    plt.xlabel("Ada step")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.grid(True)
    plt.plot(steps, accuracy, 'r')
    plt.show()


def draw_animation():
    scatters_list = glob.glob(ANIMATION_ROOT + FILE + "/*")
    clip = mpy.ImageSequenceClip(scatters_list, fps=50, durations=500)
    clip.write_gif(ANIMATION_ROOT + FILE + '/' + FILE + ".gif")


dataset = Dataset(pd.read_csv(open(FILE + ".csv")).values)
ada = AdaBoost(ITERATIONS, DecisionTreeClassifier(max_depth=1))
draw_plot(ada.boost(dataset, ANIMATION_ROOT + FILE if ANIMATE else None))
if ANIMATE:
    draw_animation()
