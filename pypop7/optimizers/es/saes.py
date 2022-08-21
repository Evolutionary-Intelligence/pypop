import numpy as np

from pypop7.optimizers.es.es import ES


class SAES(ES):
    """Self-Adaptation Evolution Strategy (SAES).

    .. note:: `SAES` adapts only the *global* step-size on-the-fly with *small* populations, often resulting in
       slow (and even premature) convergence for large-scale black-box optimization (LSBBO), especially on
       ill-conditioned optimization problems).

       It is **highly recommended** to first attempt other more advanced ES variants for LSBBO. Here we include
       it mainly for *benchmarking* and *theoretical* purpose.

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
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`),
                * 'record_fitness'           - flag to record fitness list to output results (`bool`, default: `False`),
                * 'record_fitness_frequency' - function evaluations frequency of recording (`int`, default: `1000`),

                  * if `record_fitness` is set to `False`, it will be ignored,
                  * if `record_fitness` is set to `True` and it is set to 1, all fitness generated during optimization
                    will be saved into output results.

                * 'verbose'                  - flag to print verbose info during optimization (`bool`, default: `True`),
                * 'verbose_frequency'        - frequency of printing verbose info (`int`, default: `10`);
              and with five particular settings (`keys`):
                * 'mean'          - initial (starting) point, mean of Gaussian search distribution (`array_like`),
                * 'sigma'         - initial global step-size (σ), mutation strength (`float`),
                * 'n_individuals' - number of offspring (λ: lambda), offspring population size (`int`, default:
                  `4 + int(3*np.log(self.ndim_problem))`),
                * 'n_parents'     - number of parents (μ: mu), parental population size (`int`, default:
                  `int(self.n_individuals / 2)`),
                * 'eta_sigma'     - learning rate of global step-size (`float`, default:
                  `1 / np.sqrt(2*self.ndim_problem)`).

    Examples
    --------
    Use the ES optimizer `SAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.saes import SAES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3 * numpy.ones((2,)),
       ...            'sigma': 0.1}
       >>> saes = SAES(problem, options)  # initialize the optimizer class
       >>> results = saes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"SAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
         * Generation 10: best_so_far_y 1.16832e+00, min(y) 1.52815e+01 & Evaluations 60
         * Generation 20: best_so_far_y 7.96885e-02, min(y) 8.59800e+00 & Evaluations 120
         ...
       SAES: 5000, 0.07968852575335955

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring (λ: lambda), offspring population size.
    n_parents     : `int`
                    number of parents (μ: mu), parental population size.
    mean          : `array_like`
                    initial (starting) point, mean of Gaussian search distribution.
    sigma         : `float`
                    initial global step-size (σ), mutation strength (`float`).
    eta_sigma     : `float`
                    learning rate of global step-size.

    References
    ----------
    Beyer, H.G., 2020, July.
    Design principles for matrix adaptation evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation Companion (pp. 682-700). ACM.
    https://dl.acm.org/doi/abs/10.1145/3377929.3389870

    http://www.scholarpedia.org/article/Evolution_strategies

    https://homepages.fhv.at/hgb/downloads/mu_mu_I_lambda-ES.oct
    (See its official Matlab/Octave version.)
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        if self.eta_sigma is None:
            self.eta_sigma = 1 / np.sqrt(2*self.ndim_problem)

    def initialize(self, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        sigmas = np.ones((self.n_individuals,))  # step-sizes for all offspring
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return x, mean, sigmas, y

    def iterate(self, x=None, mean=None, sigmas=None, y=None, args=None):
        for k in range(self.n_individuals):  # sample offspring population
            if self._check_terminations():
                return x, sigmas, y
            sigmas[k] = self.sigma*np.exp(self.eta_sigma*self.rng_optimization.standard_normal())
            x[k] = mean + sigmas[k]*self.rng_optimization.standard_normal((self.ndim_problem,))
            y[k] = self._evaluate_fitness(x[k], args)
        return x, sigmas, y

    def _restart_initialize(self):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self.n_restart += 1
            self.sigma = np.copy(self._sigma_bak)
            self.n_individuals *= 2
            self.n_parents = int(self.n_individuals / 2)
            self._n_generations = 0
            self._fitness_list = [np.Inf]
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
            if self.record_fitness:
                fitness.extend(y.tolist())
            if self._check_terminations():
                break
            order = np.argsort(y)[:self.n_parents]
            mean = np.mean(x[order], axis=0)
            self.sigma = np.mean(sigmas[order])
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                x, mean, sigmas, y = self.restart_initialize(x, mean, sigmas, y)
        return self._collect_results(fitness, mean)
