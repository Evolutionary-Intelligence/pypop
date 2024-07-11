import numpy as np  # engine for numerical computing

from pypop7.optimizers.es.es import ES  # abstract class of all evolution strategies (ES)


class SAES(ES):
    """Self-Adaptation Evolution Strategy (SAES).

    .. note:: `SAES` adapts only the *global* step-size on-the-fly with a *relatively small* population, often
       resulting in *slow* (and even *premature*) convergence for large-scale black-box optimization (LBO),
       especially on *ill-conditioned* fitness landscapes. Therefore, it is recommended to first attempt more
       advanced ES variants (e.g. `LMCMA`, `LMMAES`) for LBO. Here we include `SAES` mainly for *benchmarking*
       and *theoretical* purpose.

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
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.inf`),
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

    Examples
    --------
    Use the black-box optimizer `SAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.saes import SAES
       >>> problem = {'fitness_function': rosenbrock,  # to define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5.0*numpy.ones((2,)),
       ...            'upper_boundary': 5.0*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # to set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3.0*numpy.ones((2,)),
       ...            'sigma': 3.0}  # global step-size may need to be tuned
       >>> saes = SAES(problem, options)  # to initialize the optimizer class
       >>> results = saes.optimize()  # to run the optimization/evolution process
       >>> # to return the number of function evaluations and the best-so-far fitness
       >>> print(f"SAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       SAES: 5000, 0.012622712890954227

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/mvkspst4>`_ for more details.

    Attributes
    ----------
    best_so_far_x  : `array_like`
                     final best-so-far solution found during entire optimization.
    best_so_far_y  : `array_like`
                     final best-so-far fitness found during entire optimization.
    lr_sigma       : `float`
                     learning rate of global step-size adaptation.
    mean           : `array_like`
                     initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals  : `int`
                     number of offspring, aka offspring population size.
    n_parents      : `int`
                     number of parents, aka parental population size.
    sigma          : `float`
                     final global step-size, aka mutation strength (changed during optimization).

    References
    ----------
    Beyer, H.G., 2020, July.
    `Design principles for matrix adaptation evolution strategies.
    <https://dl.acm.org/doi/abs/10.1145/3377929.3389870>`_
    In Proceedings of ACM Conference on Genetic and Evolutionary Computation Companion (pp. 682-700). ACM.

    http://www.scholarpedia.org/article/Evolution_strategies

    See its official Matlab/Octave version from `Prof. Beyer <https://homepages.fhv.at/hgb/>`_:
    https://homepages.fhv.at/hgb/downloads/mu_mu_I_lambda-ES.oct
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        if self.lr_sigma is None:
            self.lr_sigma = 1.0/np.sqrt(2*self.ndim_problem)

    def initialize(self, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        sigmas = np.ones((self.n_individuals,))  # global step-sizes for all offspring
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        self._list_initial_mean.append(np.copy(mean))
        return x, mean, sigmas, y

    def iterate(self, x=None, mean=None, sigmas=None, y=None, args=None):
        for k in range(self.n_individuals):  # to sample offspring population
            if self._check_terminations():
                return x, sigmas, y
            sigmas[k] = self.sigma*np.exp(self.lr_sigma*self.rng_optimization.standard_normal())
            x[k] = mean + sigmas[k]*self.rng_optimization.standard_normal((self.ndim_problem,))
            y[k] = self._evaluate_fitness(x[k], args)
        return x, sigmas, y

    def restart_reinitialize(self, y):
        min_y = np.min(y)
        if min_y < self._list_fitness[-1]:
            self._list_fitness.append(min_y)
        else:
            self._list_fitness.append(self._list_fitness[-1])
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._list_fitness) >= self.stagnation:
            is_restart_2 = (self._list_fitness[-self.stagnation] - self._list_fitness[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self._print_verbose_info([], y, True)
            if self.verbose:
                print(' ....... *** restart *** .......')
            self._n_restart += 1
            self._list_generations.append(self._n_generations)  # for each restart
            self._n_generations = 0
            self.n_individuals *= 2
            self.n_parents = int(self.n_individuals/2)
            self._list_fitness = [np.inf]
        return is_restart

    def restart_initialize(self, x=None, mean=None, sigmas=None, y=None):
        if self.restart_reinitialize(y):
            self.sigma = np.copy(self._sigma_bak)
            x, mean, sigmas, y = self.initialize(True)
        return x, mean, sigmas, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, sigmas, y = self.initialize()
        while True:
            # sample and evaluate offspring population
            x, sigmas, y = self.iterate(x, mean, sigmas, y, args)
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            order = np.argsort(y)[:self.n_parents]
            # use intermediate multi-recombination
            mean = np.mean(x[order], axis=0)
            self.sigma = np.mean(sigmas[order])
            if self.is_restart:
                x, mean, sigmas, y = self.restart_initialize(x, mean, sigmas, y)
        return self._collect(fitness, y, mean)
