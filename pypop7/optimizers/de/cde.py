import numpy as np

from pypop7.optimizers.de.de import DE


class CDE(DE):
    """Classic Differential Evolution (CDE).

    .. note:: Typically, `DE/rand/1/bin` is seen as the **classic/basic** version of `DE`.
       `CDE` often optimizes on relatively low-dimensional (e.g., < 100) search spaces. 

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
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation (RNG) needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default: `100`),
                * 'f'             - mutation factor (`float`, default: `0.5`),
                * 'cr'            - crossover probability (`float`, default: `0.9`).

    Examples
    --------
    Use the optimizer `CDE` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.de.cde import CDE
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 0}
       >>> cde = CDE(problem, options)  # initialize the optimizer class
       >>> results = cde.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CDE: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CDE: 5000, 7.517670268794657e-09

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/3fc826yt>`_ for more details.

    Attributes
    ----------
    cr            : `float`
                    crossover probability.
    f             : `float`
                    mutation factor.
    n_individuals : `int`
                    number of offspring, aka offspring population size.

    See `User Guide <https://pypop.readthedocs.io/en/latest/user-guide.html>`_ for more options/settings.

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

        v = np.empty((self.n_individuals, self.ndim_problem))
        return x, y, v

    def mutate(self, x=None, v=None):
        n = self.n_individuals
        for i in range(self.n_individuals):
            r0 = self.rng_optimization.choice([j for j in range(n) if j != i])
            r1 = self.rng_optimization.choice([j for j in range(n) if (j != i and j != r0)])
            r2 = self.rng_optimization.choice([j for j in range(n) if (j != i and j != r0 and j != r1)])
            v[i] = x[r0] + self.f*(x[r1] - x[r2])
        return v

    def crossover(self, v=None, x=None):
        """Binomial crossover (uniform discrete crossover)."""
        for i in range(self.n_individuals):
            j_r = self.rng_optimization.integers(self.ndim_problem)
            tmp = v[i, j_r]  # to avoid loop (a trick)
            co = self.rng_optimization.random(self.ndim_problem) > self.cr
            v[i, co] = x[i, co]  # to avoid loop (a trick)
            v[i, j_r] = tmp  # to avoid loop (a trick)
        return v

    def select(self, v=None, x=None, y=None, args=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            yy = self._evaluate_fitness(v[i], args)
            if yy < y[i]:
                x[i], y[i] = v[i], yy
        return x, y

    def iterate(self, x=None, y=None, v=None, args=None):
        v = self.mutate(x, v)
        v = self.crossover(v, x)
        x, y = self.select(v, x, y, args)
        self._n_generations += 1
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = DE.optimize(self, fitness_function)
        x, y, v = self.initialize(args)
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            x, y = self.iterate(x, y, v, args)
        return self._collect(fitness, y)
