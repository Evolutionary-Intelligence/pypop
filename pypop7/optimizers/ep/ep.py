import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class EP(Optimizer):
    """Evolutionary Programming(EP)
        Reference
        -----------
        X. Yao, Y, Liu, G. Lin
        Evolutionary Programming Made Faster
        IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 3, NO. 2, JULY 1999
        https://ieeexplore.ieee.org/abstract/document/771163
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:
            self.n_individuals = 100
        if self.n_parents is None:
            self.n_parents = 10
        self._n_generations = 0
        self.mean = options.get('mean')  # mean of Gaussian search distribution
        if self.mean is None:  # 'mean' has priority over 'x'
            self.mean = options.get('x')

    def initialize(self, is_restart=False):
        raise NotImplementedError

    def iterate(self, mean, x, y):
        raise NotImplementedError

    def _initialize_mean(self, is_restart=False):
        if is_restart or (self.mean is None):
            mean = self.rng_initialization.uniform(self.initial_lower_boundary,
                                                   self.initial_upper_boundary)
        else:
            mean = np.copy(self.mean)
        return mean

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose_frequency):
            best_so_far_y = -self.best_so_far_y if self._is_maximization else self.best_so_far_y
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect_results(self, fitness, mean=None):
        results = Optimizer._collect_results(self, fitness)
        results['_n_generations'] = self._n_generations
        return results