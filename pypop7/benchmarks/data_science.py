import numpy as np

from pypop7.benchmarks.base_functions import BaseFunction


def logistic_regression(w, x, y):
    """"Cross-Entropy Loss Function of Logistic Regression (with binary labels {0, 1}).

        Note that this loss function is convex.

        https://web.stanford.edu/~jurafsky/slp3/5.pdf
        https://developer.ibm.com/articles/implementing-logistic-regression-from-scratch-in-python/
        https://jermwatt.github.io/machine_learning_refined/notes/6_Linear_twoclass_classification/6_2_Cross_entropy.html
    ""
    loss = np.empty(len(y))
    for i in range(len(y)):
        p = 1.0/(1.0 + np.exp(-np.dot(x[i], w)))
        loss[i] = -y[i]*np.log(p) - (1.0 - y[i])*np.log(1.0 - p))
    return np.mean(loss)


class LogisticRegression(BaseFunction):
    def __call__(self, w, x, y):
        return logistic_regression(w, x, y)
