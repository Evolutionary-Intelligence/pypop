import numpy as np  # engine for numerical computing

from pypop7.benchmarks.base_functions import BaseFunction


def cross_entropy_loss_lr(w, x, y):
    """Cross-Entropy Loss Function of Logistic Regression (with binary labels/classes {0, 1}).

        Note that this loss function for binary classification is convex.

        https://web.stanford.edu/~jurafsky/slp3/5.pdf
        https://developer.ibm.com/articles/implementing-logistic-regression-from-scratch-in-python/
        https://jermwatt.github.io/machine_learning_refined/notes/6_Linear_twoclass_classification/6_2_Cross_entropy.html
        https://openreview.net/forum?id=BJe-DsC5Fm (2019)
    """
    loss = np.empty(len(y))
    for i in range(len(y)):
        p = 1.0/(1.0 + np.exp(-(w[0] + np.dot(x[i], w[1:]))))
        loss[i] = -y[i]*np.log(p) - (1.0 - y[i])*np.log(1.0 - p)
    return np.mean(loss)


class CrossEntropyLossLR(BaseFunction):
    def __call__(self, w, x, y):
        return cross_entropy_loss_lr(w, x, y)


def square_loss_lr(w, x, y):
    """Square Loss Function of Logistic Regression (with binary labels/classes {0, 1}).

        Note that this loss function for binary classification is non-convex (non-linear least squares).

        https://epubs.siam.org/doi/abs/10.1137/17M1154679?journalCode=sjope8
        https://openreview.net/forum?id=BJe-DsC5Fm (2019)
        https://openreview.net/forum?id=ryxz8CVYDH
        https://epubs.siam.org/doi/abs/10.1137/1.9781611976236.23
    """
    loss = np.empty(len(y))
    for i in range(len(y)):
        loss[i] = np.square(y[i] - 1.0/(1.0 + np.exp(-(w[0] + np.dot(x[i], w[1:])))))
    return np.mean(loss)


class SquareLossLR(BaseFunction):
    def __call__(self, w, x, y):
        return square_loss_lr(w, x, y)


def logistic_loss_lr(w, x, y):
    """Logistic Loss Function of Logistic Regression (with binary labels/classes {-1, 1}).

        https://www.tandfonline.com/doi/full/10.1080/00031305.2021.2006781
    """
    loss = np.empty(len(y))
    for i in range(len(y)):
        loss[i] = np.log(1.0 + np.exp(-y[i]*(w[0] + np.dot(x[i], w[1:]))))
    return np.mean(loss)


class LogisticLossLR(BaseFunction):
    def __call__(self, w, x, y):
        return logistic_loss_lr(w, x, y)


def logistic_loss_l2(w, x, y):
    """Logistic Loss Function with L2-Regularization of Logistic Regression (with binary labels/classes {-1, 1}).

        https://epubs.siam.org/doi/abs/10.1137/17M1154679?journalCode=sjope8
    """
    return logistic_loss_lr(w, x, y) + np.sum(np.square(w))/(2.0*len(y))


class LogisticLossL2(BaseFunction):
    def __call__(self, w, x, y):
        return logistic_loss_l2(w, x, y)
