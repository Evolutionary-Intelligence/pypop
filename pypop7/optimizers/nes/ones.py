import numpy as np
from numpy.linalg import pinv

from pypop7.optimizers.nes.nes import NES


class ONES(NES):
    """Original Natural Evolution Strategy (ONES).

    .. note:: `NES` constitutes a well-principled approach to real-valued black box function optimization with
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
        options['n_individuals'] = options.get('n_individuals', 100)
        options['sigma'] = np.Inf  # not used for `SGES`
        NES.__init__(self, problem, options)
        if self.lr_mean is None:
            self.lr_mean = 1.0
        assert self.lr_mean > 0, f'`self.lr_mean` = {self.lr_mean}, but should > 0.'
        if self.lr_sigma is None:
            self.lr_sigma = 0.01
        assert self.lr_sigma > 0, f'`self.lr_sigma` = {self.lr_sigma}, but should > 0.'
        self._d_cv = np.eye(self.ndim_problem)

    def initialize(self, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        cv = np.eye(self.ndim_problem)  # covariance matrix of Gaussian search distribution
        return x, y, mean, cv

    def iterate(self, x=None, y=None, mean=None, cv=None, args=None):
        inv_cv = np.linalg.inv(cv)  # inverse of covariance matrix
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y, mean, cv, inv_cv
            x[k] = mean + np.dot(np.transpose(self._d_cv), self.rng_optimization.standard_normal((self.ndim_problem,)))
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y, mean, cv, inv_cv

    def _update_distribution(self, x=None, y=None, mean=None, inv_cv=None):
        grad_mean = np.zeros((self.n_individuals, self.ndim_problem))  # gradients of mean
        grad_cv = np.zeros((self.n_individuals, self.ndim_problem * self.ndim_problem))
        for k in range(self.n_individuals):
            diff = x[k] - mean
            grad_mean[k] = np.dot(inv_cv, diff)
            _grad_cv = 0.5*(np.dot(np.dot(inv_cv, np.outer(diff, diff)), inv_cv) - inv_cv)
            grad_cv[k] = np.ravel(np.dot(self._d_cv, _grad_cv + np.transpose(_grad_cv)))
        _grad = np.hstack((np.hstack((grad_mean, grad_cv)), np.ones((self.n_individuals, 1))))
        grad = np.dot(pinv(_grad), self._u[np.argsort(y)])[:-1]
        mean -= self.lr_mean*grad[:self.ndim_problem]
        self._d_cv -= self.lr_sigma*(grad[self.ndim_problem:]).reshape((self.ndim_problem, self.ndim_problem))
        cv = np.dot(np.transpose(self._d_cv), self._d_cv)
        return x, y, mean, cv

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = NES.optimize(self, fitness_function)
        x, y, mean, cv = self.initialize()
        while not self._check_terminations():
            # sample and evaluate offspring population
            x, y, mean, cv, inv_cv = self.iterate(x, y, mean, cv, args)
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            x, y, mean, cv = self._update_distribution(x, y, mean, inv_cv)
        return self._collect_results(fitness, mean, y)
