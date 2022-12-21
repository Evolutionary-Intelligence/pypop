import numpy as np

from pypop7.optimizers.es.es import ES


class DSAES(ES):
    """Derandomized Self-Adaptation Evolution Strategy (DSAES).

    .. note:: `DSAES` adapts all the *individual* step-sizes on-the-fly with a *relatively small* population.
       The default setting (i.e., using a `small` population) may result in *relatively fast* (local) convergence,
       but with the risk of getting trapped in suboptima on multi-modal fitness landscape.

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

                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default: `10`),
                * 'lr_sigma'      - learning rate of global step-size self-adaptation (`float`, default: `1.0/3.0`).

    Examples
    --------
    Use the optimizer `DSAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.dsaes import DSAES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> dsaes = DSAES(problem, options)  # initialize the optimizer class
       >>> results = dsaes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"DSAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       DSAES: 5000, 0.04805047881994932

    Attributes
    ----------
    lr_sigma      : `float`
                    learning rate of global step-size self-adaptation.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    sigma         : `float`
                    final global step-size, aka mutation strength.

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
        if self.lr_sigma is None:  # learning rate of global step-size adaptation
            self.lr_sigma = 1.0/3.0
        assert self.lr_sigma > 0, f'`self.lr_sigma` = {self.lr_sigma}, but should > 0.'
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
        for k in range(self.n_individuals):  # sample offspring population
            if self._check_terminations():
                return x, sigmas, y
            sigma = self.lr_sigma*self.rng_optimization.standard_normal()
            z = self.rng_optimization.standard_normal((self.ndim_problem,))
            x[k] = mean + np.exp(sigma)*self._axis_sigmas*z
            # mimick the effect of intermediate recombination
            sigmas_1 = np.power(np.exp(np.abs(z)/self._e_hnd - 1.0), 1.0/self.ndim_problem)
            sigmas_2 = np.power(np.exp(sigma), 1.0/np.sqrt(self.ndim_problem))
            sigmas[k] = self._axis_sigmas*sigmas_1*sigmas_2
            y[k] = self._evaluate_fitness(x[k], args)
        return x, sigmas, y

    def restart_reinitialize(self, x=None, mean=None, sigmas=None, y=None):
        self._fitness_list.extend(y)
        is_restart_1, is_restart_2 = np.all(self._axis_sigmas < self.sigma_threshold), False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (np.max(self._fitness_list[-self.stagnation:]) -
                            np.min(self._fitness_list[-self.stagnation:])) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self._print_verbose_info([], y, True)
            self._n_restart += 1
            self.n_individuals *= 2
            self._n_generations = 0
            self._fitness_list = [np.Inf]
            x, mean, sigmas, y = self.initialize(True)
            if self.verbose:
                print(' ....... restart .......')
        return x, mean, sigmas, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, sigmas, y = self.initialize()
        while not self._check_terminations():
            # sample and evaluate offspring population
            x, sigmas, y = self.iterate(x, mean, sigmas, y, args)
            order = np.argsort(y)[0]
            self._axis_sigmas = np.copy(sigmas[order])
            mean = np.copy(x[order])
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            if self.is_restart:
                x, mean, sigmas, y = self.restart_reinitialize(x, mean, sigmas, y)
        results = self._collect_results(fitness, mean, y)
        results['_axis_sigmas'] = self._axis_sigmas
        return results
