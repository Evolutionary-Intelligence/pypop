import numpy as np

from pypop7.optimizers.cem.cem import CEM
import torch
from lml import LML


class DCEM(CEM):
    """Differentiable Cross-Entropy Method
    Reference
    ---------------
    B. Amos, D. Yarats
    The Differentiable Cross-Entropy Method
    Proceedings of the 37th International Conference on Machine Learning, PMLR 119:291-302, 2020.
    http://proceedings.mlr.press/v119/amos20a.html
    """
    def __init__(self, problem, options):
        CEM.__init__(self, problem, options)
        self.lml_verbose = options.get('lml_verbose')
        self.lml_eps = options.get('lml_eps')

    def initialize(self, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        mean = self._initialize_mean(is_restart)
        return mean, x, y

    def iterate(self, mean, x, y):
        for i in range(self.n_individuals):
            x[i] = self.rng_optimization.normal(mean, self.sigma, self.ndim_problem)
            y[i] = self._evaluate_fitness(x[i])
        return x, y

    def update_distribution(self, x, y):
        mean_y = np.mean(y)
        std_y = np.std(y)
        _y = (y - mean_y) / (std_y + 1e-10)
        _y = _y.reshape(self.n_individuals, 1)
        _y = torch.from_numpy(_y)
        I = LML(N=self.n_parents, verbose=self.lml_verbose, eps=self.lml_eps)(_y)
        I = I.numpy()
        x_I = I * x
        mean = np.mean(x_I, axis=0)
        # mean = np.clip(mean, self.lower_boundary, self.upper_boundary)
        self.sigma = np.sqrt(np.mean(I * (x - mean)**2, axis=0))
        return mean

    def optimize(self, fitness_function=None):
        fitness = CEM.optimize(self, fitness_function)
        mean, x, y = self.initialize()
        while True:
            x, y = self.iterate(mean, x, y)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            mean = self.update_distribution(x, y)
        results = self._collect_results(fitness, mean)
        return results
