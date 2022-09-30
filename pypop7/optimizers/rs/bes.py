import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer
from pypop7.optimizers.rs.rs import RS


class BES(RS):
    """BErnoulli Smoothing (BES).

    References
    ----------
    Gao, K. and Sener, O., 2022, June.
    Generalizing Gaussian Smoothing for Random Search.
    In International Conference on Machine Learning (pp. 7077-7101). PMLR.
    https://proceedings.mlr.press/v162/gao22f.html
    https://icml.cc/media/icml-2022/Slides/16434.pdf
    """
    def __init__(self, problem, options):
        RS.__init__(self, problem, options)
        self.n_individuals = options.get('n_individuals', 100)  # number of individuals (samples)
        self.lr = options.get('lr', 0.001)  # learning rate
        self.c = options.get('c', 0.1)  # factor of finite-difference gradient estimate
        self.verbose = options.get('verbose', 10)

    def initialize(self, is_restart=False):
        if is_restart or (self.x is None):
            x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        else:
            x = np.copy(self.x)
        return x

    def iterate(self, args=None, x=None, fitness=None):  # for each iteration (generation)
        gradient = np.zeros((self.ndim_problem,))  # estimated gradient
        y_base = self._evaluate_fitness(x, args)  # for finite-difference gradient estimate
        if self.saving_fitness:
            fitness.append(y_base)
        self._print_verbose_info(y_base)
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x
            # set directional gradient based on Bernoulli distribution
            dg = self.rng_optimization.binomial(n=1, p=0.5, size=(self.ndim_problem,))
            dg = (dg - 0.5)/0.5
            y[i] = self._evaluate_fitness(x + self.c*dg, args)
            gradient += (y[i] - y_base)*dg
            if self.saving_fitness:
                fitness.append(y[i])
        gradient /= (self.c*self.n_individuals)
        x -= self.lr*gradient  # stochastic gradient descent (SGD)
        self._n_generations += 1
        self._print_verbose_info(y)
        return x

    def optimize(self, fitness_function=None, args=None):  # for all iterations (generations)
        fitness = Optimizer.optimize(self, fitness_function)
        x = self.initialize()
        while not self._check_terminations():
            x = self.iterate(args, x, fitness)
        return self._collect_results(fitness)
