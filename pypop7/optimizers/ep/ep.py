import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class EP(Optimizer):
    """Evolutionary Programming (EP).

    References
    ----------
    Yao, X., Liu, Y. and Lin, G., 1999.
    Evolutionary programming made faster.
    IEEE Transactions on Evolutionary Computation, 3(2), pp.82-102.
    https://ieeexplore.ieee.org/abstract/document/771163
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:
            self.n_individuals = 200
        if self.n_parents is None:
            self.n_parents = 100
        self._n_generations = 0

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose_frequency):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect_results(self, fitness=None):
        results = Optimizer._collect_results(self, fitness)
        results['_n_generations'] = self._n_generations
        return results
