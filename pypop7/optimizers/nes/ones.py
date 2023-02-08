import numpy as np

from pypop7.optimizers.nes.nes import NES
from pypop7.optimizers.nes.sges import SGES


class ONES(SGES):
    """Original Natural Evolution Strategy (ONES).

    .. note:: `NES` constitutes a **well-principled** approach to real-valued black box function optimization with
       a relatively clean derivation **from first principles**. Here we include `ONES` **mainly** for *benchmarking*
       and *theoretical* purpose.

    References
    ----------
    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014.
    Natural evolution strategies.
    Journal of Machine Learning Research, 15(1), pp.949-980.
    https://jmlr.org/papers/v15/wierstra14a.html

    Schaul, T., 2011.
    Studies in continuous black-box optimization.
    Doctoral Dissertation, Technische Universität München.
    https://people.idsia.ch/~schaul/publications/thesis.pdf

    See the official Python source code from PyBrain:
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/nes.py
    """
    def __init__(self, problem, options):
        SGES.__init__(self, problem, options)

    def _update_distribution(self, x=None, y=None, mean=None, cv=None):
        inv_cv = np.linalg.inv(cv)  # inverse of covariance matrix
        grad_mean = np.zeros((self.n_individuals, self.ndim_problem))  # gradients of mean
        grad_cv = np.zeros((self.n_individuals, self.ndim_problem*self.ndim_problem))
        for k in range(self.n_individuals):
            diff = x[k] - mean
            grad_mean[k] = np.dot(inv_cv, diff)
            _grad_cv = 0.5*(np.dot(np.dot(inv_cv, np.outer(diff, diff)), inv_cv) - inv_cv)
            grad_cv[k] = np.ravel(np.dot(self._d_cv, _grad_cv + _grad_cv.T))
        _grad = np.hstack((np.hstack((grad_mean, grad_cv)), np.ones((self.n_individuals, 1))))
        grad = np.dot(np.linalg.pinv(_grad), self._u[np.argsort(y)])[:-1]
        mean += self.lr_mean*grad[:self.ndim_problem]
        self._d_cv += self.lr_sigma*(grad[self.ndim_problem:].reshape((self.ndim_problem, self.ndim_problem)))
        cv = np.dot(self._d_cv.T, self._d_cv)
        return x, y, mean, cv

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = NES.optimize(self, fitness_function)
        x, y, mean, cv = self.initialize()
        while True:
            # sample and evaluate offspring population
            x, y, mean = self.iterate(x, y, mean, args)
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            x, y, mean, cv = self._update_distribution(x, y, mean, cv)
            self._n_generations += 1
            x, y, mean, cv = self.restart_reinitialize(x, y, mean, cv)
        return self._collect(fitness, y, mean)
