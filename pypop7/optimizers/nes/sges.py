import numpy as np

from pypop7.optimizers.nes.nes import NES


class SGES(NES):
    """Search Gradient-based Evolution Strategy (SGES).

    .. note:: Here we include it **only** for *theoretical* and/or *educational* purpose.

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
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/ves.py
    """
    def __init__(self, problem, options):
        options['n_individuals'] = options.get('n_individuals', 100)
        options['sigma'] = np.Inf  # not used for `SGES`
        NES.__init__(self, problem, options)
        if self.lr_mean is None:
            self.lr_mean = 0.01
        assert self.lr_mean > 0.0, f'`self.lr_mean` = {self.lr_mean}, but should > 0.0.'
        if self.lr_sigma is None:
            self.lr_sigma = 0.01
        assert self.lr_sigma > 0.0, f'`self.lr_sigma` = {self.lr_sigma}, but should > 0.0.'
        self._d_cv = None

    def initialize(self, is_restart=False):
        NES.initialize(self)
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        cv = np.eye(self.ndim_problem)  # covariance matrix of Gaussian search distribution
        self._d_cv = np.eye(self.ndim_problem)
        return x, y, mean, cv

    def iterate(self, x=None, y=None, mean=None, args=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y, mean
            x[k] = mean + np.dot(self._d_cv.T, self.rng_optimization.standard_normal((self.ndim_problem,)))
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y, mean

    def _update_distribution(self, x=None, y=None, mean=None, cv=None):
        order = np.argsort(y)
        inv_cv = np.linalg.inv(cv)  # inverse of covariance matrix
        grad_mean = np.zeros((self.ndim_problem,))  # gradients of mean
        grad_cv = np.zeros((self.ndim_problem, self.ndim_problem))  # gradient of covariance matrix
        for k in range(self.n_individuals):
            diff = x[k] - mean
            grad_mean += self._u[order[k]]*np.dot(inv_cv, diff)
            _grad_cv = 0.5*(np.dot(np.dot(inv_cv, np.outer(diff, diff)), inv_cv) - inv_cv)
            grad_cv += self._u[order[k]]*np.dot(self._d_cv, _grad_cv + _grad_cv.T)
        mean += self.lr_mean*grad_mean/self.n_individuals
        self._d_cv += self.lr_sigma*grad_cv/self.n_individuals
        cv = np.dot(self._d_cv.T, self._d_cv)
        return x, y, mean, cv

    def restart_reinitialize(self, x=None, y=None, mean=None, cv=None):
        if self.is_restart and NES.restart_reinitialize(self, y):
            x, y, mean, cv = self.initialize(True)
        return x, y, mean, cv

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = NES.optimize(self, fitness_function)
        x, y, mean, cv = self.initialize()
        while True:
            x, y, mean = self.iterate(x, y, mean, args)
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            x, y, mean, cv = self._update_distribution(x, y, mean, cv)
            self._n_generations += 1
            x, y, mean, cv = self.restart_reinitialize(x, y, mean, cv)
        return self._collect(fitness, y, mean)
