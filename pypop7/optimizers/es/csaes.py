import numpy as np

from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.dsaes import DSAES


class CSAES(DSAES):
    """Cumulative Step-size Adaptation Evolution Strategy (CSAES).

    .. note:: `CSAES` adapts the *individual* step-sizes on-the-fly with *small* populations, according to the
       well-known `Cumulative Step-size Adaptation (CSA) <http://link.springer.com/chapter/10.1007/3-540-58484-6_263>`_
       rule from the evolutionary computation community.

       AKA cumulative path length control.

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
                * 'max_runtime'              - maximal runtime (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'mean'          - initial (starting) point, mean of Gaussian search distribution (`array_like`),
                * 'sigma'         - initial global step-size (σ), mutation strength (`float`),
                * 'n_individuals' - number of offspring (λ: lambda), offspring population size (`int`, default:
                  `4 + int(np.floor(3*np.log(problem.get('ndim_problem'))))`),
                * 'n_parents'     - number of parents (μ: mu), parental population size (`int`, default:
                  `int(options['n_individuals']/4)`),
                * 'lr_sigma'     - learning rate of global step-size (`float`, default:
                  `np.sqrt(options['n_parents']/(problem['ndim_problem'] + options['n_parents']))`).

    Examples
    --------
    Use the ES optimizer `CSAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.csaes import CSAES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3 * numpy.ones((2,)),
       ...            'sigma': 0.1}
       >>> csaes = CSAES(problem, options)  # initialize the optimizer class
       >>> results = csaes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CSAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CSAES: 5000, 0.026416618757717083

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring (λ: lambda), offspring population size.
    n_parents     : `int`
                    number of parents (μ: mu), parental population size.
    mean          : `array_like`
                    mean of Gaussian search distribution.
    sigma         : `float`
                    mutation strength.
    lr_sigma      : `float`
                    learning rate of global step-size.

    References
    ----------
    Hansen, N., Arnold, D.V. and Auger, A., 2015.
    Evolution strategies.
    In Springer Handbook of Computational Intelligence (pp. 871-898). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007%2F978-3-662-43505-2_44
    (See Algorithm 44.6 for details.)

    Ostermeier, A., Gawelczyk, A. and Hansen, N., 1994, October.
    Step-size adaptation based on non-local use of selection information
    In International Conference on Parallel Problem Solving from Nature (pp. 189-198).
    Springer, Berlin, Heidelberg.
    http://link.springer.com/chapter/10.1007/3-540-58484-6_263
    """
    def __init__(self, problem, options):
        if options.get('n_individuals') is None:
            options['n_individuals'] = 4 + int(np.floor(3*np.log(problem.get('ndim_problem'))))
        if options.get('n_parents') is None:
            options['n_parents'] = int(options['n_individuals']/4)
        if options.get('lr_sigma') is None:
            options['lr_sigma'] = np.sqrt(options['n_parents']/(
                    problem['ndim_problem'] + options['n_parents']))
        DSAES.__init__(self, problem, options)
        self._s_1 = None  # for Line 8
        self._s_2 = None  # for Line 8
        # E[||N(0,I)||]: expectation of chi distribution
        self._e_chi = np.sqrt(self.ndim_problem)*(
                1.0 - 1.0/(4.0*self.ndim_problem) + 1.0/(21.0*np.power(self.ndim_problem, 2)))

    def initialize(self, is_restart=False):
        self._s_1 = 1.0 - self.lr_sigma
        self._s_2 = np.sqrt(self.lr_sigma*(2.0 - self.lr_sigma)*self.n_parents)
        self._axis_sigmas = self._sigma_bak*np.ones((self.ndim_problem,))
        z = np.empty((self.n_individuals, self.ndim_problem))  # noise for offspring population
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        s = np.zeros((self.ndim_problem,))  # evolution path
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return z, x, mean, s, y

    def iterate(self, z=None, x=None, mean=None, y=None, args=None):
        # sample offspring population (Line 4)
        for k in range(self.n_individuals):
            if self._check_terminations():
                return z, x, y
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))  # Line 5
            x[k] = mean + self._axis_sigmas*z[k]  # Line 6
            y[k] = self._evaluate_fitness(x[k], args)
        return z, x, y

    def _update_distribution(self, z=None, x=None, s=None, y=None):
        order = np.argsort(y)[:self.n_parents]  # Line 7
        s = self._s_1*s + self._s_2*np.mean(z[order], axis=0)  # Line 8
        sigmas_1 = np.power(np.exp(np.abs(s)/self._e_hnd - 1.0), 1.0/(3.0*self.ndim_problem))
        sigmas_2 = np.power(np.exp(np.linalg.norm(s)/self._e_chi - 1.0),
                            self.lr_sigma/(1.0 + np.sqrt(self.n_parents/self.ndim_problem)))
        self._axis_sigmas *= (sigmas_1*sigmas_2)  # Line 9
        mean = np.mean(x[order], axis=0)  # Line 11
        return s, mean

    def restart_initialize(self, z=None, x=None, mean=None, s=None, y=None):
        if self._restart_initialize():
            self.n_parents = int(self.n_individuals/4)
            self.lr_sigma = np.sqrt(self.n_parents/(self.ndim_problem + self.n_parents))
            z, x, mean, s, y = self.initialize(True)
        return z, x, mean, s, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        z, x, mean, s, y = self.initialize()
        while True:
            # sample and evaluate offspring population
            z, x, y = self.iterate(z, x, mean, y, args)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            s, mean = self._update_distribution(z, x, s, y)
            self._print_verbose_info(y)
            self._n_generations += 1
            if self.is_restart:
                z, x, mean, s, y = self.restart_initialize(z, x, mean, s, y)
        results = self._collect_results(fitness, mean)
        results['s'] = s
        results['_axis_sigmas'] = self._axis_sigmas
        return results
