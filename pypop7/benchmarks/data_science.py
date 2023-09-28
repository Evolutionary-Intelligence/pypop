import numpy as np

from pypop7.benchmarks.base_functions import BaseFunction


def log_loss_logistics_regression(x, y, w):
    loss = []
    for xx, yy in zip(x, y):
        p = 1/(1 + np.exp(np.dot(xx, w)))
        loss.append(-yy*np.log(p) - (1 - yy)*np.log(1 - p))
    return np.mean(loss)


class LogLossLogisticsRegression(BaseFunction):
    def __call__(self, x, y, w):
        return log_loss_logistics_regression(x, y, w)
