import numpy as np

from optimizers.es.es import ES


class NES(ES):
    """Natural Evolution Strategies (NES).

    Reference
    ---------
    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014.
    Natural evolution strategies.
    Journal of Machine Learning Research, 15(27), pp.949-980.
    https://jmlr.org/papers/v15/wierstra14a.html
    """
    def __init__(self, problem, options):
        problem['_is_maximization'] = True  # mandatory setting for NES
        ES.__init__(self, problem, options)

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def optimize(self, fitness_function=None):
        raise NotImplementedError

    def _compute_gradients(self):
        raise NotImplementedError

    def _update_distribution(self):
        raise NotImplementedError

    def _fitness_shaping(self):
        base = np.log(self.n_individuals / 2 + 1)
        utilities = np.maximum(0, [base - np.log(k) for k in (np.arange(self.n_individuals) + 1)])
        return utilities / np.sum(utilities) - (1 / self.n_individuals)
