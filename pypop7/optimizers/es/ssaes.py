import numpy as np

from pypop7.optimizers.es.es import ES


class SSAES(ES):
    """Schwefel's Self-Adaptation Evolution Strategy (SSAES).

    .. note:: `SSAES` adapts the *individual* step-sizes on-the-fly, proposed by Schwefel. Since it needs the
       *large* populations (e.g. larger than number of dimensions) for reliable adaptation, `SSAES` often
       suffers from *slow* convergence for large-scale black-box optimization (LSBBO).

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
              and with six particular settings (`keys`):
                * 'mean'            - initial (starting) point, mean of Gaussian search distribution (`array_like`),
                * 'sigma'           - initial global step-size (σ), mutation strength (`float`),
                * 'n_individuals'   - number of offspring (λ: lambda), offspring population size (`int`, default:
                  `5*problem['ndim_problem']`),
                * 'n_parents'       - number of parents (μ: mu), parental population size (`int`, default:
                  `int(options['n_individuals'] / 4)`),
                * 'eta_sigma'       - learning rate of global step-size (`float`, default:
                  `1.0 / np.sqrt(problem['ndim_problem']`),
                * 'eta_axis_sigmas' - learning rate of individual step-sizes (`float`, default:
                  `1.0 / np.power(problem['ndim_problem'], 1.0 / 4.0)`).

    Examples
    --------
    Use the ES optimizer `SSAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.ssaes import SSAES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3 * numpy.ones((2,)),
       ...            'sigma': 0.1}
       >>> ssaes = SSAES(problem, options)  # initialize the optimizer class
       >>> results = ssaes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"SSAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
         * Generation 10: best_so_far_y 2.14285e-01, min(y) 2.14285e-01 & Evaluations 100
         * Generation 20: best_so_far_y 2.10049e-01, min(y) 2.10462e-01 & Evaluations 200
         ...
         * Generation 480: best_so_far_y 1.33198e-01, min(y) 1.33198e-01 & Evaluations 4800
         * Generation 490: best_so_far_y 1.31319e-01, min(y) 1.31645e-01 & Evaluations 4900
       SSAES: 5000, 0.13131869620062903

    Attributes
    ----------
    n_individuals   : `int`
                      number of offspring (λ: lambda), offspring population size.
    n_parents       : `int`
                      number of parents (μ: mu), parental population size.
    mean            : `array_like`
                      initial (starting) point, mean of Gaussian search distribution.
    sigma           : `float`
                      initial global step-size (σ), mutation strength (`float`).
    eta_sigma       : `float`
                      learning rate of global step-size.
    eta_axis_sigmas : `float`
                      learning rate of individual step-sizes.

    References
    ----------
    Hansen, N., Arnold, D.V. and Auger, A., 2015.
    Evolution strategies.
    In Springer Handbook of Computational Intelligence (pp. 871-898). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007%2F978-3-662-43505-2_44
    """
    def __init__(self, problem, options):
        if options.get('n_individuals') is None:
            options['n_individuals'] = 5*problem.get('ndim_problem')
        if options.get('n_parents') is None:
            options['n_parents'] = int(options['n_individuals'] / 4)
        ES.__init__(self, problem, options)
        self.axis_sigmas = self.sigma*np.ones((self.ndim_problem,))  # individual step-sizes
        if self.eta_sigma is None:
            self.eta_sigma = 1.0 / np.sqrt(self.ndim_problem)  # learning rate of global step-size
        assert self.eta_sigma > 0, f'`self.eta_sigma` = {self.eta_sigma}, but should > 0.'
        # learning rate of individual step-sizes
        self.eta_axis_sigmas = options.get('eta_axis_sigmas', 1.0 / np.power(self.ndim_problem, 1.0 / 4.0))

    def initialize(self):
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean()  # mean of Gaussian search distribution
        sigmas = np.empty((self.n_individuals, self.ndim_problem))  # individual step-sizes for all offspring
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return x, mean, sigmas, y

    def iterate(self, x=None, mean=None, sigmas=None, y=None, args=None):
        for k in range(self.n_individuals):  # sample population
            if self._check_terminations():
                return x, sigmas, y
            sigma = self.eta_sigma*self.rng_optimization.standard_normal()
            axis_sigmas = self.eta_axis_sigmas*self.rng_optimization.standard_normal((self.ndim_problem,))
            sigmas[k] = self.axis_sigmas*np.exp(axis_sigmas)*np.exp(sigma)
            x[k] = mean + sigmas[k]*self.rng_optimization.standard_normal((self.ndim_problem,))
            y[k] = self._evaluate_fitness(x[k], args)
        return x, sigmas, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, sigmas, y = self.initialize()
        while True:
            # sample and evaluate offspring population
            x, sigmas, y = self.iterate(x, mean, sigmas, y, args)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            order = np.argsort(y)[:self.n_parents]
            self.axis_sigmas = np.mean(sigmas[order], axis=0)
            mean = np.mean(x[order], axis=0)
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness, mean)
        results['axis_sigmas'] = self.axis_sigmas
        return results
