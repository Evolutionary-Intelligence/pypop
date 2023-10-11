import numpy as np  # engine for numerical computing

from pypop7.optimizers.nes.ones import ONES


def _combine_block(ll):
    ll = [list(map(np.mat, row)) for row in ll]
    h = [m.shape[1] for m in ll[0]]
    v, v_i = [row[0].shape[0] for row in ll], 0
    mat = np.zeros((sum(h), sum(v)))
    for i, row in enumerate(ll):
        h_i = 0
        for j, m in enumerate(row):
            mat[v_i:v_i + v[i], h_i:h_i + h[j]] = m
            h_i += h[j]
        v_i += v[i]
    return mat


class ENES(ONES):
    """Exact Natural Evolution Strategy (ENES).

    Parameters
    ----------
    problem : dict
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : dict
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - number of offspring/descendants, aka offspring population size (`int`),
                * 'n_parents'     - number of parents/ancestors, aka parental population size (`int`),
                * 'mean'          - initial (starting) point (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'sigma'         - initial global step-size, aka mutation strength (`float`),
                * 'lr_mean'       - learning rate of distribution mean update (`float`, default: `1.0`),
                * 'lr_sigma'      - learning rate of global step-size adaptation (`float`, default: `1.0`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.nes.enes import ENES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> enes = ENES(problem, options)  # initialize the optimizer class
       >>> results = enes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"ENES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       ENES: 5000, 0.00035668252927080496

    Attributes
    ----------
    lr_mean       : `float`
                    learning rate of distribution mean update.
    lr_sigma      : `float`
                    learning rate of global step-size adaptation.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search/sampling/mutation distribution.
    n_individuals : `int`
                    number of offspring/descendants, aka offspring population size.
    n_parents     : `int`
                    number of parents/ancestors, aka parental population size.
    sigma         : `float`
                    global step-size, aka mutation strength (i.e., overall std of Gaussian search distribution).

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
    In International Conference on Machine Learning (pp. 1161-1168). ACM.
    https://dl.acm.org/doi/abs/10.1145/1553374.1553522

    See the official Python source code from PyBrain:
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/nes.py
    """
    def __init__(self, problem, options):
        ONES.__init__(self, problem, options)
        if options.get('lr_mean') is None:
            self.lr_mean = 1.0
        if options.get('lr_sigma') is None:
            self.lr_sigma = 1.0

    def _update_distribution(self, x=None, y=None, mean=None, cv=None):
        order = np.argsort(-y)
        u = np.empty((self.n_individuals,))
        for i, o in enumerate(order):
            u[o] = self._u[i]
        inv_d, inv_cv = np.linalg.inv(self._d_cv), np.linalg.inv(cv)
        dd = np.diag(np.diag(inv_d))
        v = np.zeros((self._n_distribution, self.n_individuals))
        for k in range(self.n_individuals):
            diff = x[k] - mean
            s = np.dot(inv_d.T, diff)
            v[:self.ndim_problem, k] += diff
            v[self.ndim_problem:, k] += self._triu2flat(np.outer(s, np.dot(inv_d, s)) - dd)
        j = self._n_distribution - 1
        g = 1.0/(inv_cv[-1, -1] + np.square(inv_d[-1, -1]))
        d = 1.0/inv_cv[-1, -1]
        v[j, :] = np.dot(g, v[j, :])
        j -= 1
        for k in reversed(list(range(self.ndim_problem - 1))):
            w = inv_cv[k, k]
            w_g = w + np.square(inv_d[k, k])
            q = np.dot(d, inv_cv[k + 1:, k])
            c = np.dot(inv_cv[k + 1:, k], q)
            r, r_g = 1.0/(w - c), 1.0/(w_g - c)
            t, t_g = -(1.0 + r*c)/w, -(1.0 + r_g*c)/w_g
            g = _combine_block([[r_g, t_g*q], [np.mat(t_g*q).T, d + r_g*np.outer(q, q)]])
            d = _combine_block([[r, t*q], [np.mat(t*q).T, d + r*np.outer(q, q)]])
            v[j - (self.ndim_problem - k - 1):j + 1, :] = np.dot(g, v[j - (self.ndim_problem - k - 1):j + 1, :])
            j -= self.ndim_problem - k
        grad, v_2 = np.zeros((self._n_distribution,)), v*v
        j = self._n_distribution - 1
        for k in reversed(list(range(self.ndim_problem))):
            base = np.sum(v_2[j - (self.ndim_problem - k - 1):j + 1, :], 0)
            grad[j - (self.ndim_problem - k - 1):j + 1] = np.dot(
                v[j - (self.ndim_problem - k - 1):j + 1, :], u - np.dot(base, u)/np.sum(base))
            j -= self.ndim_problem - k
        base = np.sum(v_2[:j + 1, :], 0)
        grad[:j + 1] = np.dot(v[:j + 1, :], (u - np.dot(base, u)/np.sum(base)))
        grad /= self.n_individuals
        mean += self.lr_mean*grad[:self.ndim_problem]
        self._d_cv += self.lr_sigma*self._flat2triu(grad[self.ndim_problem:])
        cv = np.dot(self._d_cv.T, self._d_cv)
        return x, y, mean, cv
