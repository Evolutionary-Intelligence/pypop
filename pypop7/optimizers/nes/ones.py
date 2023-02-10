import numpy as np

from pypop7.optimizers.nes.sges import SGES


class ONES(SGES):
    """Original Natural Evolution Strategy (ONES).

    .. note:: `NES` constitutes a **well-principled** approach to real-valued black box function optimization with
       a relatively clean derivation **from first principles**. Here we include `ONES` **mainly** for *benchmarking*
       and *theoretical* purpose.

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

    See the official Python source code from PyBrain:
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/nes.py
    """
    def __init__(self, problem, options):
        SGES.__init__(self, problem, options)
        if options.get('lr_mean') is None:
            self.lr_mean = 1.0
        if options.get('lr_sigma') is None:
            self.lr_sigma = 1.0

    def _update_distribution(self, x=None, y=None, mean=None, cv=None):
        order = np.argsort(-y)
        u = np.empty((self.n_individuals,))
        for i, o in enumerate(order):
            u[o] = self._u[i]
        inv_cv = np.linalg.inv(cv)
        phi = np.ones((self.n_individuals, self._n_distribution + 1))
        for k in range(self.n_individuals):
            diff = x[k] - mean
            phi[k, :self.ndim_problem] = np.dot(inv_cv, diff)
            _grad_cv = 0.5*(np.dot(np.dot(inv_cv, np.outer(diff, diff)), inv_cv) - inv_cv)
            phi[k, self.ndim_problem:-1] = self._triu2flat(np.dot(self._d_cv, _grad_cv + _grad_cv.T))
        grad = np.dot(np.linalg.pinv(phi), u)[:-1]
        mean += self.lr_mean*grad[:self.ndim_problem]
        self._d_cv += self.lr_sigma*self._flat2triu(grad[self.ndim_problem:])
        cv = np.dot(self._d_cv.T, self._d_cv)
        return x, y, mean, cv
