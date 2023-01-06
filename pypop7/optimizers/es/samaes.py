import time

import numpy as np

from pypop7.optimizers.es.saes import SAES


class SAMAES(SAES):
    """A Simple Recombinative MA-ES with Self-Adaptation (SAMAES).

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
                * 'lr_sigma'      - learning rate of global step-size (`float`, default:
                  `1.0/np.sqrt(2*problem['ndim_problem'])`).
                * 'lr_matrix'     - learning rate of matrix (`float`, default:
                  `1.0/(2 + ((problem['ndim_problem'] + 1)*problem['ndim_problem'])/options['n_parents'])`).

    Attributes
    ----------
    lr_sigma      : `float`
                    learning rate of global step-size.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.
    sigma         : `float`
                    final global step-size, aka mutation strength.
    lr_matrix     : `float`
                    learning rate of matrix

    References
    ----------
    Beyer, H.G., 2020, July.
    Design principles for matrix adaptation evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation Companion (pp. 682-700). ACM.
    https://dl.acm.org/doi/abs/10.1145/3377929.3389870
    """
    def __init__(self, problem, options):
        SAES.__init__(self, problem, options)
        self.lr_matrix = 1.0/(2 + ((self.ndim_problem + 1)*self.ndim_problem)/self.n_parents)

    def initialize(self, is_restart=False):
        x, mean, sigmas, y = SAES.initialize(self, is_restart=False)
        m = np.eye(self.ndim_problem)  # M matrix
        return x, mean, sigmas, y, m

    def iterate(self, x=None, mean=None, sigmas=None, y=None, m=None, args=None):
        z = np.empty((self.n_individuals, self.ndim_problem))
        for k in range(self.n_individuals):  # to sample offspring population
            if self._check_terminations():
                return x, sigmas, y, m, z
            sigmas[k] = self.sigma*np.exp(self.lr_sigma*self.rng_optimization.standard_normal())
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            x[k] = mean + sigmas[k]*np.matmul(m, z[k])
            y[k] = self._evaluate_fitness(x[k], args)
        return x, sigmas, y, m, z

    def restart_initialize(self, x=None, mean=None, sigmas=None, y=None, m=None):
        if self._restart_initialize(y):
            self.sigma = np.copy(self._sigma_bak)
            x, mean, sigmas, y, m = self.initialize(True)
        return x, mean, sigmas, y, m

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        fitness = []  # to store all fitness generated during evolution/optimization

        x, mean, sigmas, y, m = self.initialize()
        while not self._check_terminations():
            # sample and evaluate offspring population
            x, sigmas, y, m, z = self.iterate(x, mean, sigmas, y, m, args)
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            order = np.argsort(y)[:self.n_parents]
            # use intermediate multi-recombination
            mean = np.mean(x[order], axis=0)
            self.sigma = np.mean(sigmas[order])
            zz = np.zeros((self.ndim_problem, self.ndim_problem))
            for i in z[order]:
                zz += np.tile(i[:,None], (1, self.ndim_problem))*np.tile(i, (self.ndim_problem, 1))
            m *= (np.eye(self.ndim_problem) + self.lr_matrix * (zz/self.n_parents - np.eye(self.ndim_problem)))
            if self.is_restart:
                x, mean, sigmas, y, m = self.restart_initialize(x, mean, sigmas, y)
        return self._collect(fitness, y, mean)
