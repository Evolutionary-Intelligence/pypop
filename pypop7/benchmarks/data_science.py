import numpy as np

from pypop7.benchmarks.base_functions import BaseFunction


def logistics_regression(x, w):
    y = 1/(1 + np.exp(-np.dot(x, w)))
    return y


class Sphere(BaseFunction):
    def __call__(self, x, w):
        return logistics_regression(x, w)
