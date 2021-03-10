import numpy as np


def fitness_shaping(y):
    n_individuals = y.size
    tmp = np.log(n_individuals / 2 + 1)
    tmp = [max(0, tmp - np.log(k)) for k in range(1, n_individuals + 1)]
    return tmp / np.sum(tmp) - (1 / n_individuals)
