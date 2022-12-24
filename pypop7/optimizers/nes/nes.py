import numpy as np

from pypop7.optimizers.es.es import ES


class NES(ES):
    """Natural Evolution Strategies (NES).

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

    Yi, S., Wierstra, D., Schaul, T. and Schmidhuber, J., 2009, June.
    Stochastic search using the natural gradient.
    In Proceedings of International Conference on Machine Learning (pp. 1161-1168).
    https://dl.acm.org/doi/10.1145/1553374.1553522

    Wierstra, D., Schaul, T., Peters, J. and Schmidhuber, J., 2008, June.
    Natural evolution strategies.
    In IEEE Congress on Evolutionary Computation (pp. 3381-3387). IEEE.
    https://ieeexplore.ieee.org/abstract/document/4631255
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        _u = np.maximum(0.0, np.log(self.n_individuals/2 + 1) - np.log(np.arange(self.n_individuals) + 1))
        self._u = _u/np.sum(_u) - 1/self.n_individuals

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError
