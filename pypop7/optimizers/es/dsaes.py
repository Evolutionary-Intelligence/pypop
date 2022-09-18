import numpy as np

from pypop7.optimizers.es.es import ES


class DSAES(ES):
    """Derandomized Self-Adaptation Evolution Strategy (DSAES).

    .. note:: `DSAES` adapts the *individual* step-sizes on-the-fly with *small* populations. To obtain
       satisfactory performance for large-scale black-box optimization, the number of offspring may
       need to be carefully tuned.

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
                * 'n_individuals' - number of offspring (λ: lambda), offspring population size (`int`, default: `10`),
                * 'lr_sigma'      - learning rate of global step-size (`float`, default: `1.0/3.0`).

    Examples
    --------
    Use the ES optimizer `DSAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.dsaes import DSAES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3 * numpy.ones((2,)),
       ...            'sigma': 0.1}
       >>> dsaes = DSAES(problem, options)  # initialize the optimizer class
       >>> results = dsaes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"DSAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       DSAES: 5000, 0.04805047881994932

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring (λ: lambda), offspring population size.
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
    (See Algorithm 44.5 for details.)

    Ostermeier, A., Gawelczyk, A. and Hansen, N., 1994.
    A derandomized approach to self-adaptation of evolution strategies.
    Evolutionary Computation, 2(4), pp.369-380.
    https://direct.mit.edu/evco/article-abstract/2/4/369/1407/A-Derandomized-Approach-to-Self-Adaptation-of
    """
    def __init__(self, problem, options):
        if options.get('n_individuals') is None:
            options['n_individuals'] = 10
        ES.__init__(self, problem, options)
        if self.lr_sigma is None:
            self.lr_sigma = 1.0/3.0
        self._axis_sigmas = None
        self._e_hnd = np.sqrt(2.0/np.pi)  # E[|N(0,1)|]: expectation of half-normal distribution

    def initialize(self, is_restart=False):
        self._axis_sigmas = self._sigma_bak*np.ones((self.ndim_problem,))
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        # set individual step-sizes for all offspring
        sigmas = np.ones((self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return x, mean, sigmas, y

    def iterate(self, x=None, mean=None, sigmas=None, y=None, args=None):
        # sample offspring population (Line 4)
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, sigmas, y
            sigma = self.lr_sigma*self.rng_optimization.standard_normal()  # Line 5
            z = self.rng_optimization.standard_normal((self.ndim_problem,))  # Line 6
            x[k] = mean + np.exp(sigma)*self._axis_sigmas*z  # Line 7
            # mimick the effect of intermediate recombination (Line 8)
            sigmas_1 = np.power(np.exp(np.abs(z)/self._e_hnd - 1.0), 1.0/self.ndim_problem)
            sigmas_2 = np.power(np.exp(sigma), 1.0/np.sqrt(self.ndim_problem))
            sigmas[k] = self._axis_sigmas*sigmas_1*sigmas_2
            y[k] = self._evaluate_fitness(x[k], args)
        return x, sigmas, y

    def _restart_initialize(self):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2 = np.all(self._axis_sigmas < self.sigma_threshold), False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self._n_restart += 1
            self.n_individuals *= 2
            self._n_generations = 0
            self._fitness_list = [np.Inf]
            if self.verbose:
                print(' ...*... restart ...*...')
        return is_restart

    def restart_initialize(self, x=None, mean=None, sigmas=None, y=None):
        if self._restart_initialize():
            x, mean, sigmas, y = self.initialize(True)
        return x, mean, sigmas, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, sigmas, y = self.initialize()
        while True:
            # sample and evaluate offspring population
            x, sigmas, y = self.iterate(x, mean, sigmas, y, args)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            order = np.argsort(y)[0]  # Line 9
            self._axis_sigmas = np.copy(sigmas[order])  # Line 10
            mean = np.copy(x[order])  # Line 11
            self._print_verbose_info(y)
            self._n_generations += 1
            if self.is_restart:
                x, mean, sigmas, y = self.restart_initialize(x, mean, sigmas, y)
        results = self._collect_results(fitness, mean)
        results['_axis_sigmas'] = self._axis_sigmas
        return results
