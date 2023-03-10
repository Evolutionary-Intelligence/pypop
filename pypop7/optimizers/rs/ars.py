import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer
from pypop7.optimizers.rs.rs import RS


class ARS(RS):
    """
    Accelerated random search(ARS).

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
                * 'c'                        - contraction factor (`float`, default: `np.sqrt(2)`),
                * 'L'                        - restart time (`int`, default: `4`),
                * 'p'                        - precision threshold (`float`, default: `1e-4`),
                * 'max_r'                    - maximum of radius (`float`, default: `1.0`),
                * 'x'                        - initial (starting) point (`array_like`),
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.rs.ars import ARS
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'L': 4,
       ...            'c': 2022,
       ...            'p': 1e-4,
       ...            'max_r': 5,
       ...            'seed_rng': 2022}
       >>> ars = ARS(problem, options)  # initialize the optimizer class
       >>> results = ars.optimize()  # run the optimization process
       >>> # return the number of used function evaluations and found best-so-far fitness
       >>> print(f"ARS: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       ARS: 5000, 0.08943097577742222

    Attributes
    ----------
    c             : `float`
                    contraction factor.
    L             : `int`
                    restart time
    p             : `float`
                    precision threshold.
    max_r          : `float`
                    maximum radius.
    x             : `array_like`
                    initial (starting) point.

    References
    ----------
    Appel, M.J., Labarre, R. and Radulovic, D., 2004.
    On accelerated random search.
    SIAM Journal on Optimization, 14(3), pp.708-731.
    https://epubs.siam.org/doi/abs/10.1137/S105262340240063X
    """
    def __init__(self, problem, options):
        RS.__init__(self, problem, options)
        self.L = options.get('L', 4)  # restart time
        self.c = options.get('c', np.sqrt(2))  # contraction factor
        self.p = options.get('p', 1e-4)  # precision threshold
        self.max_r = options.get('max_r', 1.0)  # maximum radius

    def initialize(self, args=None):
        n_restart = 0
        r = self.max_r
        x = self._sample(self.rng_initialization)
        y = self._evaluate_fitness(x, args)
        return n_restart, r, x, y

    def _sample(self, rng):
        x = rng.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        return x

    def iterate(self, n_restart, r, x, y):
        if n_restart > self.L:
            r = self.max_r
            n_restart = 0
            x = self._sample(self.rng_optimization)
            y = self._evaluate_fitness(x)
        else:
            r_n = np.sqrt(r / self.ndim_problem)
            x1 = x + self.rng_optimization.uniform(-1 * r_n, r_n)
            y1 = self._evaluate_fitness(x1)
            if y1 < y:
                x = x1
                r = self.max_r
                y = y1
            else:
                r /= self.c
                if r < self.p:
                    r = self.max_r
                    n_restart += 1
        return n_restart, r, x, y

    def optimize(self, fitness_function=None, args=None):  # for all iterations (generations)
        fitness = Optimizer.optimize(self, fitness_function)
        n_restart, r, x, y = self.initialize()  # starting point
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            n_restart, r, x, y = self.iterate(n_restart, r, x, y)  # to sample a new point
            self._n_generations += 1
        return self._collect(fitness, y)
