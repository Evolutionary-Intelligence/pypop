import numpy as np

from pypop7.optimizers.de.de import DE


class TDE(DE):
    """Trigonometric-mutation Differential Evolution (TDE).

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
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default: `30`),
                * 'f'             - mutation factor (`float`, default: `0.99`),
                * 'cr'            - crossover probability (`float`, default: `0.85`),
                * 'tm'            - trigonometric mutation probability (`float`, default: `0.05`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.de.tde import TDE
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 0}
       >>> tde = TDE(problem, options)  # initialize the optimizer class
       >>> results = tde.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"TDE: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       TDE: 5000, 6.420787226215637e-21

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/56frubv2>`_ for more details.

    Attributes
    ----------
    cr            : `float`
                    crossover probability.
    f             : `float`
                    mutation factor.
    tm            : 'float
                    trigonometric mutation probability.
    n_individuals : `int`
                    number of offspring, aka offspring population size.

    References
    ----------
    Fan, H.Y. and Lampinen, J., 2003.
    A trigonometric mutation operation to differential evolution.
    Journal of Global Optimization, 27(1), pp.105-129.
    https://link.springer.com/article/10.1023/A:1024653025686
    """
    def __init__(self, problem, options):
        DE.__init__(self, problem, options)
        self.n_individuals = options.get('n_individuals', 30)  # population size
        self.f = options.get('f', 0.99)  # mutation factor
        self.tm = options.get('tm', 0.05)  # trigonometric mutation probability
        self.cr = options.get('cr', 0.85)  # crossover probability

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # population
        y = np.empty((self.n_individuals,))  # fitness
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y

    def mutate(self, x=None, y=None):
        v = np.empty((self.n_individuals, self.ndim_problem))
        for i in range(self.n_individuals):
            r = self.rng_optimization.permutation(self.n_individuals)[:4]
            r = r[r != i][:3]  # a simple yet effective trick
            if self.rng_optimization.random() < self.tm:  # trigonometric mutation
                p = np.abs(y[r[0]]) + np.abs(y[r[1]]) + np.abs(y[r[2]])
                p1, p2, p3 = np.abs(y[r[0]])/p, np.abs(y[r[1]])/p, np.abs(y[r[2]])/p
                v[i] = ((x[r[0]] + x[r[1]] + x[r[2]])/3.0 + (p2 - p1)*(x[r[0]] - x[r[1]]) +
                        (p3 - p2)*(x[r[1]] - x[r[2]]) + (p1 - p3)*(x[r[2]] - x[r[0]]))
            else:  # same as the original DE version (`DE/rand/1/bin`)
                v[i] = x[r[0]] + self.f*(x[r[1]] - x[r[2]])
        return v

    def crossover(self, v=None, x=None):
        for i in range(self.n_individuals):
            k = self.rng_optimization.integers(self.ndim_problem)
            for j in range(self.ndim_problem):
                if (self.rng_optimization.random() <= self.cr) or (k == j):
                    continue
                else:
                    v[i][j] = x[i][j]
        return v

    def select(self, v=None, x=None, y=None, args=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            yy = self._evaluate_fitness(v[i], args)
            if yy <= y[i]:
                x[i], y[i] = v[i], yy
        return x, y

    def iterate(self, x=None, y=None, args=None):
        return self.select(self.crossover(self.mutate(x, y), x), x, y, args)

    def optimize(self, fitness_function=None, args=None):
        fitness = DE.optimize(self, fitness_function)
        x, y = self.initialize(args)
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            x, y = self.iterate(x, y, args)
            self._n_generations += 1
        return self._collect(fitness, y)
