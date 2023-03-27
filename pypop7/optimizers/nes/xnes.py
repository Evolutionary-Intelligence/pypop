import numpy as np
import scipy

from pypop7.optimizers.nes.nes import NES


class XNES(NES):
    """Exponential Natural Evolution Strategies (XNES).

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

                * 'sigma'         - initial global step-size, aka mutation strength (`float`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.nes.xnes import XNES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> xnes = XNES(problem, options)  # initialize the optimizer class
       >>> results = xnes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"XNES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       XNES: 5000, 1.3565728021697798e-18

    Attributes
    ----------
    lr_cv         : `float`
                    learning rate of covariance matrix adaptation.
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

    Glasmachers, T., Schaul, T., Yi, S., Wierstra, D. and Schmidhuber, J., 2010, July.
    Exponential natural evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 393-400).
    https://dl.acm.org/doi/abs/10.1145/1830483.1830557

    See the official Python source code from PyBrain:
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/xnes.py
    """
    def __init__(self, problem, options):
        options['sigma'] = np.Inf  # not used for `SGES`
        NES.__init__(self, problem, options)
        self.lr_cv = 0.6*(3.0 + np.log(self.ndim_problem))/self.ndim_problem/np.sqrt(self.ndim_problem)
        self.lr_sigma = 0.6*(3.0 + np.log(self.ndim_problem))/self.ndim_problem/np.sqrt(self.ndim_problem)
        self._eye = np.eye(self.ndim_problem)

    def initialize(self, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        a = np.eye(self.ndim_problem)
        inv_a = np.eye(self.ndim_problem)
        log_det = 0.0
        self._w = np.maximum(0.0, np.log(self.n_individuals/2.0 + 1.0) - np.log(
            self.n_individuals - np.arange(self.n_individuals)))
        return x, y, mean, a, inv_a, log_det

    def iterate(self, x=None, y=None, mean=None, a=None, args=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            x[k] = mean + np.dot(a, self.rng_optimization.standard_normal((self.ndim_problem,)))
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _update_distribution(self, x=None, y=None, mean=None, a=None, inv_a=None, log_det=None):
        order = np.argsort(-y)
        u = np.empty((self.n_individuals,))
        for i, o in enumerate(order):
            u[o] = self._w[i]
        u = u/np.sum(u) - 1.0/self.n_individuals
        s = np.empty((self.n_individuals, self.ndim_problem))
        for k in range(self.n_individuals):
            s[k] = np.dot(inv_a, x[k] - mean)
        d_c = np.dot(s.T, u)
        g_cv = np.dot(np.array([np.outer(k, k) - self._eye for k in s]).T, u)
        trace = np.trace(g_cv)
        g_cv -= trace/self.ndim_problem*self._eye
        d_a = 0.5*(self.lr_sigma*trace/self.ndim_problem*self._eye + self.lr_cv*g_cv)
        mean += np.dot(a, d_c)
        a = np.dot(a, scipy.linalg.expm(d_a))
        inv_a = np.dot(scipy.linalg.expm(-d_a), inv_a)
        log_det += 0.5*self.lr_sigma*trace
        return mean, a, inv_a, log_det

    def restart_reinitialize(self, x=None, y=None, mean=None, a=None, inv_a=None, log_det=None):
        if self.is_restart and NES.restart_reinitialize(self, y):
            x, y, mean, a, inv_a, log_det = self.initialize(True)
        return x, y, mean, a, inv_a, log_det

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = NES.optimize(self, fitness_function)
        x, y, mean, a, inv_a, log_det = self.initialize()
        while True:
            x, y = self.iterate(x, y, mean, a, args)
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            mean, a, inv_a, log_det = self._update_distribution(x, y, mean, a, inv_a, log_det)
            self._n_generations += 1
            x, y, mean, a, inv_a, log_det = self.restart_reinitialize(x, y, mean, a, inv_a, log_det)
        return self._collect(fitness, y, mean)
