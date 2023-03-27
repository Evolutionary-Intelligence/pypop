import numpy as np

from pypop7.optimizers.nes.nes import NES


class SGES(NES):
    """Search Gradient-based Evolution Strategy (SGES).

    .. note:: Here we include it **only** for *theoretical* and/or *educational* purpose.

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
                * 'lr_mean'       - learning rate of distribution mean update (`float`, default: `0.01`),
                * 'lr_sigma'      - learning rate of global step-size adaptation (`float`, default: `0.01`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.nes.sges import SGES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> sges = SGES(problem, options)  # initialize the optimizer class
       >>> results = sges.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"SGES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       SGES: 5000, 0.01906602832229609

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

    See the official Python source code from PyBrain:
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/ves.py
    """
    def __init__(self, problem, options):
        options['n_individuals'] = options.get('n_individuals', 100)
        options['sigma'] = np.Inf  # not used for `SGES`
        NES.__init__(self, problem, options)
        if self.lr_mean is None:
            self.lr_mean = 0.01
        assert self.lr_mean > 0.0, f'`self.lr_mean` = {self.lr_mean}, but should > 0.0.'
        if self.lr_sigma is None:
            self.lr_sigma = 0.01
        assert self.lr_sigma > 0.0, f'`self.lr_sigma` = {self.lr_sigma}, but should > 0.0.'
        self._n_distribution = int(self.ndim_problem + self.ndim_problem*(self.ndim_problem+1)/2)
        self._d_cv = None

    def initialize(self, is_restart=False):
        NES.initialize(self)
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        cv = np.eye(self.ndim_problem)  # covariance matrix of Gaussian search distribution
        self._d_cv = np.eye(self.ndim_problem)
        return x, y, mean, cv

    def iterate(self, x=None, y=None, mean=None, args=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            x[k] = mean + np.dot(self._d_cv.T, self.rng_optimization.standard_normal((self.ndim_problem,)))
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _triu2flat(self, cv):
        g = np.zeros((int(self.ndim_problem*(self.ndim_problem+1)/2),))
        s, e = 0, self.ndim_problem
        for r in range(self.ndim_problem):
            g[s:e] = cv[r, r:]
            s = e
            e += (self.ndim_problem - (r + 1))
        return g

    def _flat2triu(self, g):
        cv = np.zeros((self.ndim_problem, self.ndim_problem))
        s, e = 0, self.ndim_problem
        for r in range(self.ndim_problem):
            cv[r, r:] = g[s:e]
            s = e
            e += (self.ndim_problem - (r + 1))
        return cv

    def _update_distribution(self, x=None, y=None, mean=None, cv=None):
        order = np.argsort(-y)
        u = np.empty((self.n_individuals,))
        for i, o in enumerate(order):
            u[o] = self._u[i]
        inv_cv = np.linalg.inv(cv)
        phi = np.zeros((self.n_individuals, self._n_distribution))
        phi[:, :self.ndim_problem] = np.dot(inv_cv, (x - mean).T).T
        grad_cv = np.empty((self.n_individuals, int(self.ndim_problem*(self.ndim_problem + 1)/2)))
        for k in range(self.n_individuals):
            diff = x[k] - mean
            _grad_cv = 0.5*(np.dot(np.dot(inv_cv, np.outer(diff, diff)), inv_cv) - inv_cv)
            grad_cv[k] = self._triu2flat(np.dot(self._d_cv, (_grad_cv + _grad_cv.T)))
        phi[:, self.ndim_problem:] = grad_cv
        phi_square = phi*phi
        grad = np.sum(phi*(np.outer(u, np.ones((self._n_distribution,))) - np.dot(
            u, phi_square)/np.dot(np.ones((self.n_individuals,)), phi_square)), 0)
        mean += self.lr_mean*grad[:self.ndim_problem]
        self._d_cv += self.lr_sigma*self._flat2triu(grad[self.ndim_problem:])
        cv = np.dot(self._d_cv.T, self._d_cv)
        return mean, cv

    def restart_reinitialize(self, x=None, y=None, mean=None, cv=None):
        if self.is_restart and NES.restart_reinitialize(self, y):
            x, y, mean, cv = self.initialize(True)
        return x, y, mean, cv

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = NES.optimize(self, fitness_function)
        x, y, mean, cv = self.initialize()
        while True:
            x, y = self.iterate(x, y, mean, args)
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            mean, cv = self._update_distribution(x, y, mean, cv)
            self._n_generations += 1
            x, y, mean, cv = self.restart_reinitialize(x, y, mean, cv)
        return self._collect(fitness, y, mean)
