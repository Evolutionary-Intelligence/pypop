import numpy as np  # engine for numerical computing
from scipy.stats import multivariate_normal as mn

from pypop7.optimizers.cem import CEM


class MRAS(CEM):
    """Model Reference Adaptive Search (MRAS).

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
                * 'sigma'         - initial global step-size, aka mutation strength (`float`),
                * 'mean'          - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default: `1000`),
                * 'p'             - percentage of samples as parents (`int`, default: `0.1`),
                * 'alpha'         - increasing factor of samples/individuals (`float`, default: `1.1`),
                * 'v'             - smoothing factor for search distribution update (`float`, default: `0.2`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.cem.mras import MRAS
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'sigma': 10}  # the global step-size may need to be tuned for better performance
       >>> mras = MRAS(problem, options)  # initialize the optimizer class
       >>> results = mras.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"MRAS: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       MRAS: 5000, 0.18363570418709932

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/yv44nbwu>`_ for more details.

    Attributes
    ----------
    alpha         : `float`
                    increasing factor of samples/individuals.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    p             : `float`
                    percentage of samples as parents.
    sigma         : `float`
                    initial global step-size, aka mutation strength,
    v             : `float`
                    smoothing factor for search distribution update.

    References
    ----------
    Hu, J., Fu, M.C. and Marcus, S.I., 2007.
    A model reference adaptive search method for global optimization.
    Operations Research, 55(3), pp.549-568.
    https://pubsonline.informs.org/doi/abs/10.1287/opre.1060.0367
    """
    def __init__(self, problem, options):
        CEM.__init__(self, problem, options)
        self.p = options.get('p', 0.1)  # percentage of samples as parents
        assert 0.0 < self.p <= 1.0
        self.alpha = options.get('alpha', 1.1)  # increasing factor of samples/individuals
        assert self.alpha > 1.0
        self.v = options.get('v', 0.2)  # smoothing factor for search distribution update
        self.r = options.get('r', 1e-4)
        self.epsilon = options.get('epsilon', 1e-5)
        assert self.epsilon >= 0.0
        self._lambda = 0.01
        self._initial_pdf = None
        self._gamma = None
        self._min_elitists = 5*self.ndim_problem

    def initialize(self, is_restart=False):
        mean, cov = self._initialize_mean(is_restart), np.diag((self.sigma**2)*np.ones((self.ndim_problem,)))
        x = np.empty((self.n_individuals, self.ndim_problem))  # samples (population)
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        self._initial_pdf = [np.copy(mean), np.copy(cov)]
        return mean, cov, x, y

    def iterate(self, mean=None, cov=None, x=None, y=None, args=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return mean, cov, x, y
            # use a very simple way to generate a mixed pdf
            if self.rng_optimization.uniform() <= 1.0 - self._lambda:
                x[i] = self.rng_optimization.multivariate_normal(mean=mean, cov=cov)
            else:
                x[i] = self.rng_optimization.multivariate_normal(
                    mean=self._initial_pdf[0], cov=self._initial_pdf[1])
            y[i] = self._evaluate_fitness(x[i], args)
        order = np.argsort(y)
        gamma = y[order[int(np.ceil(self.p*self.n_individuals))]]  # fitness threshold for top-p elitists
        if self._n_generations == 0 or gamma <= self._gamma + self.epsilon/2:  # Step 3(a)
            self._gamma = gamma
        else:  # Step 3(b)
            yy, _p = np.Inf, np.Inf
            # this is not detailed in the original paper (but may not be a key setting)
            for i in np.linspace(self.p, 0, 100)[1:]:
                yy = y[order[int(np.ceil(i*self.n_individuals))]]
                if yy <= self._gamma + self.epsilon/2:
                    _p = i
                    break
            if 0 < _p < self.p:  # Step 3(c)
                self._gamma, self.p = yy, _p
            else:
                self.n_individuals = int(np.ceil(self.alpha*self.n_individuals))
        if np.sum(y <= self._gamma) > self._min_elitists:
            _mean, _cov = np.zeros((self.ndim_problem,)), np.zeros((self.ndim_problem, self.ndim_problem))
            _n, weights, pdfs = 0, np.zeros((len(y),)), np.zeros((len(y),))
            for i in range(len(y)):
                if y[i] <= self._gamma:
                    _n += 1
                    pdfs[i] = ((1.0 - self._lambda)*mn.pdf(x[i], mean=mean, cov=cov) + self._lambda *
                               mn.pdf(x[i], mean=self._initial_pdf[0], cov=self._initial_pdf[1]))
                    weights[i] = np.power(np.exp(-self.r*y[i]), self._n_generations)/pdfs[i]
                    _mean += weights[i]*x[i]
            if _n > 0:
                _mean = (_mean/_n)/(np.sum(weights)/_n)
                for i in range(len(y)):
                    if y[i] <= self._gamma:
                        xm = x[i] - _mean
                        _cov += weights[i]*np.dot(xm[:, np.newaxis], xm[np.newaxis, :])
                _cov = (_cov/_n)/(np.sum(weights)/_n)
                mean = self.v*_mean + (1.0 - self.v)*mean
                cov = self.v*_cov + (1.0 - self.v)*cov
        return mean, cov, x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = CEM.optimize(self, fitness_function)
        mean, cov, x, y = self.initialize()
        while True:
            mean, cov, x, y = self.iterate(mean, cov, x, y, args)
            self._print_verbose_info(fitness, y)
            if self._check_terminations():
                break
            self._n_generations += 1
            if self.n_individuals > len(y):
                x = np.empty((self.n_individuals, self.ndim_problem))  # samples (population)
                y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return self._collect(fitness, y, mean)
