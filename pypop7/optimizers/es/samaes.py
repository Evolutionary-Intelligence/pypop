import numpy as np

from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.saes import SAES


class SAMAES(SAES):
    """Self-Adaptation Matrix Adaptation Evolution Strategy (SAMAES).

    .. note:: It is **highly recommended** to first attempt more advanced ES variants (e.g. `LMCMA`, `LMMAES`) for
       large-scale black-box optimization. Here we include it only for *benchmarking* and *theoretical* purpose.

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

                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default:
                  `4 + int(3*np.log(problem['ndim_problem']))`),
                * 'n_parents'     - number of parents, aka parental population size (`int`, default:
                  `int(options['n_individuals']/2)`),
                * 'lr_sigma'      - learning rate of global step-size adaptation (`float`, default:
                  `1.0/np.sqrt(2*problem['ndim_problem'])`).
                * 'lr_matrix'     - learning rate of matrix adaptation (`float`, default:
                  `1.0/(2.0 + ((problem['ndim_problem'] + 1.0)*problem['ndim_problem'])/options['n_parents'])`).

    Examples
    --------
    Use the optimizer `SAMAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.samaes import SAMAES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> samaes = SAMAES(problem, options)  # initialize the optimizer class
       >>> results = samaes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"SAMAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       SAMAES: 5000, 5.674972930091687e-17

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/56k42a2n>`_ for more details.

    Attributes
    ----------
    lr_sigma      : `float`
                    learning rate of global step-size adaptation.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.
    sigma         : `float`
                    final global step-size, aka mutation strength.
    lr_matrix     : `float`
                    learning rate of matrix adaptation.

    References
    ----------
    Beyer, H.G., 2020, July.
    Design principles for matrix adaptation evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation Companion (pp. 682-700). ACM.
    https://dl.acm.org/doi/abs/10.1145/3377929.3389870
    """
    def __init__(self, problem, options):
        SAES.__init__(self, problem, options)
        if self.lr_sigma is None:
            self.lr_sigma = 1.0/np.sqrt(2.0*self.ndim_problem)
        self.lr_matrix = 1.0/(2.0 + ((self.ndim_problem + 1.0)*self.ndim_problem)/self.n_parents)
        self._eye = np.eye(self.ndim_problem)  # for matrix adaptation

    def initialize(self, is_restart=False):
        x, mean, sigmas, y = SAES.initialize(self, is_restart)
        m = np.eye(self.ndim_problem)  # for matrix adaptation
        return x, mean, sigmas, y, m

    def iterate(self, x=None, mean=None, sigmas=None, y=None, m=None, args=None):
        z = np.empty((self.n_individuals, self.ndim_problem))
        d = np.empty((self.n_individuals, self.ndim_problem))
        for k in range(self.n_individuals):  # to sample offspring population
            if self._check_terminations():
                return x, sigmas, y, m, z, d
            sigmas[k] = self.sigma*np.exp(self.lr_sigma*self.rng_optimization.standard_normal())
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            d[k] = np.matmul(m, z[k])
            x[k] = mean + sigmas[k]*d[k]
            y[k] = self._evaluate_fitness(x[k], args)
        return x, sigmas, y, m, z, d

    def restart_initialize(self, x=None, mean=None, sigmas=None, y=None, m=None):
        if self.is_restart and self._restart_initialize(y):
            x, mean, sigmas, y, m = self.initialize(True)
        return x, mean, sigmas, y, m

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, sigmas, y, m = self.initialize()
        while not self._check_terminations():
            # sample and evaluate offspring population
            x, sigmas, y, m, z, d = self.iterate(x, mean, sigmas, y, m, args)
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            order = np.argsort(y)[:self.n_parents]
            mean = np.mean(x[order], axis=0)  # intermediate multi-recombination
            self.sigma = np.mean(sigmas[order])  # intermediate multi-recombination
            # use the following code (fast version) owing to its quadratic time complexity
            dz = np.zeros((self.ndim_problem, self.ndim_problem))  # for matrix adaptation
            for i in range(self.n_parents):
                dz += np.outer(d[order[i]], z[order[i]])
            m = (1.0 - self.lr_matrix)*m + self.lr_matrix*(dz/self.n_parents)
            x, mean, sigmas, y, m = self.restart_initialize(x, mean, sigmas, y, m)
        return self._collect(fitness, y, mean)
