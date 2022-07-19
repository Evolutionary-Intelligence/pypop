import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class EDA(Optimizer):
    """Estimation of distribution algorithms(EDA)
    Reference
    --------------
    P. A. N. Bosman, D. Thierens
    Expanding from Discrete to Continuous Estimation of Distribution Algorithms: The IDEA
    PPSN 2000: 767-776
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self._n_generations = 0
        if self.n_individuals is None:  # number of offspring, offspring population size (λ: lambda)
            self.n_individuals = 4 + int(3 * np.log(self.ndim_problem))  # for small population setting
        if self.n_parents is None:  # number of parents, parental population size (μ: mu)
            self.n_parents = int(self.n_individuals / 2)

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose_frequency):
            best_so_far_y = -self.best_so_far_y if self._is_maximization else self.best_so_far_y
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, best_so_far_y, np.min(y), self.n_function_evaluations))

    def _initialize_x(self, is_restart=False):
        if is_restart or (self.x is None):
            x = self.rng_initialization.uniform(self.initial_lower_boundary,
                                                self.initial_upper_boundary)
        else:
            x = np.copy(self.x)
        return x

    def _collect_results(self, fitness, mean=None):
        results = Optimizer._collect_results(self, fitness)
        results['_n_generations'] = self._n_generations
        return results
