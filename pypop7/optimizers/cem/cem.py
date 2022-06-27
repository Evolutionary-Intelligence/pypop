import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class CEM(Optimizer):
    """Cross-Entropy Method(CE)
    Reference
    -----------
    P. T. de Boer, D. P. Kroese, S. Mannor, R. Y. Rubinstein, (2003)
    A Tutorial on the Cross-Entropy Method
    http://web.mit.edu/6.454/www/www_fall_2003/gew/CEtutorial.pdf
    T. Hoem-de-Mello, R. Y. Rubinstein,
    Estimation of Rare Event Probabilities using Cross-Entropy
    Proceedings of the 2002 Winter Simulation Conference
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1172900
    The official python version:
    https://pypi.org/project/cross-entropy-method/
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:
            self.n_individuals = 100
        if self.n_parents is None:
            self.n_parents = 10
        self._n_generations = 0
        self.sigma = np.ones((self.ndim_problem,)) * options.get('sigma')
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
        results['sigma'] = self.sigma
        results['_n_generations'] = self._n_generations
        return results
