import numpy as np

from pypop7.optimizers.de.de import DE


class CDE(DE):
    """Classic Differential Evolution (CDE).

    Parameters
    ----------
    problem : `dict`
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : `dict`
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`),
                * 'record_fitness'           - flag to record fitness list to output results (`bool`, default: `False`),
                * 'record_fitness_frequency' - function evaluations frequency of recording (`int`, default: `1000`),

                  * if `record_fitness` is set to `False`, it will be ignored,
                  * if `record_fitness` is set to `True` and it is set to 1, all fitness generated during optimization
                    will be saved into output results.

                * 'verbose'                  - flag to print verbose information during optimization (`bool`, default:
                  `True`),
                * 'verbose_frequency'        - generation frequency of printing verbose information (`int`, default:
                  `10`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - population size (`int`, default: `100`),
                * 'f'             - mutation factor (`float`, default: `0.5`),
                * 'cr'            - crossover probability (`float`, default: `0.9`).

    Examples
    --------
    Use the ES optimizer `CDE` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.de.cde import CDE
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 0}
       >>> cde = CDE(problem, options)  # initialize the optimizer class
       >>> results = cde.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CDE: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CDE: 5000, 1.0490841426568313e-10

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/3fc826yt>`_ for more details.

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring, offspring population size.
    f             : `float`
                    mutation factor.
    cr            : `float`
                    crossover probability.

    References
    ----------
    Price, K.V., 2013.
    Differential evolution.
    In Handbook of optimization (pp. 187-214). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-642-30504-7_8

    Price, K.V., Storn, R.M. and Lampinen, J.A., 2005.
    Differential evolution: A practical approach to global optimization.
    Springer Science & Business Media.
    https://link.springer.com/book/10.1007/3-540-31306-0

    Storn, R.M. and Price, K.V. 1997.
    Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces.
    Journal of Global Optimization, 11(4), pp.341–359.
    https://link.springer.com/article/10.1023/A:1008202821328
    """
    def __init__(self, problem, options):
        DE.__init__(self, problem, options)
        self.f = options.get('f', 0.5)  # mutation factor
        self.cr = options.get('cr', 0.9)  # crossover probability

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # population
        y = np.empty((self.n_individuals,))  # fitness
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        self._n_generations += 1
        return x, y

    def mutate(self, x=None):
        v, base = np.empty((self.n_individuals, self.ndim_problem)), np.arange(self.n_individuals)
        for i in range(self.n_individuals):
            r0 = self.rng_optimization.choice(np.setdiff1d(base, i))
            r1 = self.rng_optimization.choice(np.setdiff1d(base, [i, r0]))
            r2 = self.rng_optimization.choice(np.setdiff1d(base, [i, r0, r1]))
            v[i] = x[r0] + self.f*(x[r1] - x[r2])
        return v

    def crossover(self, v=None, x=None):
        """Binomial crossover (uniform discrete crossover)."""
        for i in range(self.n_individuals):
            j_rand = self.rng_optimization.integers(self.ndim_problem)
            for j in range(self.ndim_problem):
                if j == j_rand or self.rng_optimization.random() <= self.cr:
                    pass  # directly inherit from v
                else:  # directly inherit from x
                    v[i, j] = x[i, j]
        return v

    def select(self, v=None, x=None, y=None, args=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            yy = self._evaluate_fitness(v[i], args)
            if yy < y[i]:
                x[i], y[i] = v[i], yy
        return x, y

    def iterate(self, args=None, x=None, y=None):
        v = self.mutate(x)
        v = self.crossover(v, x)
        x, y = self.select(v, x, y, args)
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = DE.optimize(self, fitness_function)
        x, y = self.initialize(args)
        fitness.extend(y)
        while True:
            x, y = self.iterate(args, x, y)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        return self._collect_results(fitness)
