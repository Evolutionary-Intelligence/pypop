import numpy as np

from pypop7.benchmarks.base_functions import BaseFunction


def logistic_regression(w, x, y):
    loss = []
    for i in range(len(w)):
        p = 1.0/(1.0 + np.exp(np.dot(x[i], w)))
        loss.append(-y[i]*np.log(p) - (1 - y[i])*np.log(1 - p))
    return np.mean(loss)


class LogisticRegression(BaseFunction):
    def __call__(self, w, x, y):
        return logistic_regression(w, x, y)
