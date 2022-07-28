import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class DS(Optimizer):
    """Direct Search (DS).

    Reference
    ---------
    Hooke, R. and Jeeves, T.A., 1961.
    “Direct search” solution of numerical and statistical problems.
    Journal of the ACM, 8(2), pp.212-229.
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self.x = options.get('x')  # initial point
        self.sigma = options.get('sigma')  # global step-size
        self._n_generations = 0
        # for restart
        self.n_restart = 0
        self._sigma_bak = np.copy(self.sigma)
        self.sigma_threshold = options.get('sigma_threshold', 1e-10)
        self._fitness_list = [self.best_so_far_y]  # store `best_so_far_y` generated in each generation
        self.stagnation = options.get('stagnation', np.maximum(32, self.ndim_problem))  # number of generations
        self.fitness_diff = options.get('fitness_diff', 1e-10)  # threshold of fitness difference

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _initialize_x(self, is_restart=False):
        if is_restart or (self.x is None):
            x = self.rng_initialization.uniform(self.initial_lower_boundary,
                                                self.initial_upper_boundary)
        else:
            x = np.copy(self.x)
        return x

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose_frequency):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect_results(self, fitness):
        results = Optimizer._collect_results(self, fitness)
        results['sigma'] = self.sigma
        results['_n_generations'] = self._n_generations
        results['n_restart'] = self.n_restart
        return results
