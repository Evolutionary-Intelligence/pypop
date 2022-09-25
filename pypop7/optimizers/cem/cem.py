import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class CEM(Optimizer):
    """Cross-Entropy Method (CEM).

    This is the **base** (abstract) class for all `CEM` classes. Please use any of its instantiated subclasses to
    optimize the black-box problem at hand.

    .. note:: `CEM` is a class of well-established, principled population-based optimizers, proposed originally
       by Rubinstein, whose core idea is based on Kullback–Leibler (or cross-entropy) minimization.

    References
    ----------
    Kroese, D.P., Porotsky, S. and Rubinstein, R.Y., 2006.
    The cross-entropy method for continuous multi-extremal optimization.
    Methodology and Computing in Applied Probability, 8(3), pp.383-407.
    https://link.springer.com/article/10.1007/s11009-006-9753-0

    De Boer, P.T., Kroese, D.P., Mannor, S. and Rubinstein, R.Y., 2005.
    A tutorial on the cross-entropy method.
    Annals of Operations Research, 134(1), pp.19-67.
    https://link.springer.com/article/10.1007/s10479-005-5724-z

    Rubinstein, R.Y. and Kroese, D.P., 2004.
    The cross-entropy method: a unified approach to combinatorial optimization,
        Monte-Carlo simulation, and machine learning.
    New York: Springer.
    https://link.springer.com/book/10.1007/978-1-4757-4321-0
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # number of individuals (samples)
            self.n_individuals = 1000
        if self.n_parents is None:  # number of elitists
            self.n_parents = 200
        self.mean = options.get('mean')  # mean of Gaussian search (sampling/mutation) distribution
        if self.mean is None:
            self.mean = options.get('x')
        self.sigma = options.get('sigma')  # global (overall) step-size
        assert self.sigma is not None
        self._sigmas = self.sigma*np.ones((self.ndim_problem,))  # individual step-sizes
        self._n_generations = 0

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _initialize_mean(self, is_restart=False):
        if is_restart or (self.mean is None):
            mean = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        else:
            mean = np.copy(self.mean)
        return mean

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect_results(self, fitness, mean=None):
        results = Optimizer._collect_results(self, fitness)
        results['mean'] = mean
        results['_sigmas'] = self._sigmas
        results['_n_generations'] = self._n_generations
        return results
