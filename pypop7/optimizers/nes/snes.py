import numpy as np

from pypop7.optimizers.nes.nes import NES


class SNES(NES):
    """Separable Natural Evolution Strategies (SNES).

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

    Schaul, T., Glasmachers, T. and Schmidhuber, J., 2011, July.
    High dimensions and heavy tails for natural evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 845-852). ACM.
    https://dl.acm.org/doi/abs/10.1145/2001576.2001692

    See the official Python source code from PyBrain:
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/snes.py
    """
    def __init__(self, problem, options):
        NES.__init__(self, problem, options)
        self.lr_cv = 0.6*(3.0 + np.log(self.ndim_problem))/3.0/np.sqrt(self.ndim_problem)

    def initialize(self, is_restart=False):
        s = np.empty((self.n_individuals, self.ndim_problem))  # noise of offspring population
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        d = self.sigma*np.ones((self.ndim_problem,))  # individual step-sizes
        self._w = np.maximum(0.0, np.log(self.n_individuals/2.0 + 1.0) - np.log(
            self.n_individuals - np.arange(self.n_individuals)))
        return s, y, mean, d

    def iterate(self, s=None, y=None, mean=None, d=None, args=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return s, y
            s[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            y[k] = self._evaluate_fitness(mean + d*s[k], args)
        return s, y

    def _update_distribution(self, s=None, y=None, mean=None, d=None):
        order = np.argsort(-y)
        u = np.empty((self.n_individuals,))
        for i, o in enumerate(order):
            u[o] = self._w[i]
        u = u/np.sum(u) - 1.0/self.n_individuals
        mean += d*np.dot(u, s)
        d *= np.exp(0.5*self.lr_cv*np.dot(u, [np.square(k) - 1.0 for k in s]))
        return mean, d

    def restart_reinitialize(self, s=None, y=None, mean=None, d=None):
        if self.is_restart and NES.restart_reinitialize(self, y):
            s, y, mean, d = self.initialize(True)
        return s, y, mean, d

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = NES.optimize(self, fitness_function)
        s, y, mean, d = self.initialize()
        while True:
            s, y = self.iterate(s, y, mean, d, args)
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            mean, d = self._update_distribution(s, y, mean, d)
            self._n_generations += 1
            s, y, mean, d = self.restart_reinitialize(s, y, mean, d)
        return self._collect(fitness, y, mean)
