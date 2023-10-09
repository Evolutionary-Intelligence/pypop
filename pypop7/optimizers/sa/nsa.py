import numpy as np  # engine for numerical computing

from pypop7.optimizers.core.optimizer import Optimizer
from pypop7.optimizers.sa.sa import SA


class NSA(SA):
    """Noisy Simulated Annealing (NSA).

    .. note:: This is a *slightly modified* version of discrete `NSA` for continuous optimization.

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
                * 'x'         - initial (starting) point (`array_like`),
                * 'sigma'     - initial global step-size (`float`),
                * 'is_noisy'  - whether or not to minimize a **noisy** cost function (`bool`, default: `False`),
                * 'schedule'  - schedule for sampling intensity (`str`, default: `linear`),

                  * currently only two (*linear* or *quadratic*) schedules are supported for sampling intensity,

                * 'n_samples' - number of samples (`int`),
                * 'rt'        - reducing factor of annealing temperature (`float`, default: `0.99`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.sa.nsa import NSA
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3*numpy.ones((2,)),
       ...            'sigma': 1.0,
       ...            'temperature': 100.0}
       >>> nsa = NSA(problem, options)  # initialize the optimizer class
       >>> results = nsa.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"NSA: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       NSA: 5000, 0.006086567926462302

    For its correctness checking of coding, the *code-based repeatability report* cannot be provided owing to
    the lack of some details of its experiments in the original paper.

    Attributes
    ----------
    is_noisy  : `bool`
                whether or not to minimize a noisy cost function.
    n_samples : `int`
                number of samples for each iteration.
    rt        : `float`
                reducing factor of annealing temperature.
    schedule  : `str`
                schedule for sampling intensity.
    sigma     : `float`
                global step-size (fixed during optimization).
    x         : `array_like`
                initial (starting) point.

    References
    ----------
    Bouttier, C. and Gavra, I., 2019.
    Convergence rate of a simulated annealing algorithm with noisy observations.
    Journal of Machine Learning Research, 20(1), pp.127-171.
    https://www.jmlr.org/papers/v20/16-588.html
    """
    def __init__(self, problem, options):
        SA.__init__(self, problem, options)
        self.sigma = options.get('sigma')
        assert self.sigma > 0.0
        self.is_noisy = options.get('is_noisy', False)
        assert self.is_noisy in [True, False]
        self.schedule = options.get('schedule', 'linear')  # schedule for sampling intensity
        assert self.schedule in ['linear', 'quadratic'],\
            'Currently only two (*linear* or *quadratic*) schedules are supported for sampling intensity.'
        if self.is_noisy:
            self.n_samples = options.get('n_samples')
        else:
            self.n_samples = 1  # a mandatory setting
        assert self.n_samples > 0
        self.rt = options.get('cr', 0.99)  # reducing factor of temperature
        self._tk = 0

    def initialize(self, args=None):
        if self.x is None:  # starting point
            x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        else:
            x = np.copy(self.x)
        assert len(x) == self.ndim_problem
        y = self._evaluate_fitness(x, args)
        self.parent_x, self.parent_y = np.copy(x), np.copy(y)
        return y

    def iterate(self, args=None):
        x = self.parent_x + self.sigma*self.rng_optimization.standard_normal((self.ndim_problem,))
        self._tk += self.rng_optimization.exponential()
        if self.schedule == 'linear':
            n_tk = self._tk
        else:  # quadratic
            n_tk = np.square(self._tk)
        if self.n_samples is None:
            n_samples = self.rng_optimization.poisson(n_tk) + 1
        else:
            n_samples = self.n_samples
        y, parent_y = [], []
        for _ in range(n_samples):
            if self._check_terminations():
                break
            y.append(self._evaluate_fitness(x, args))
        if self.is_noisy:  # for noisy optimization
            for _ in range(n_samples):
                if self._check_terminations():
                    break
                parent_y.append(self._evaluate_fitness(self.parent_x, args))
        else:  # for static optimization
            parent_y = self.parent_y
        diff = np.mean(parent_y) - np.mean(y)
        if (diff >= 0) or (self.rng_optimization.random() < np.exp(diff/self.temperature)):
            self.parent_x, self.parent_y = np.copy(x), np.copy(y)
        if not self.is_noisy:  # for static optimization
            parent_y = []
        if len(parent_y) > 0:
            y.extend(parent_y)
        return y

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        y = self.initialize(args)
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            y = self.iterate(args)
            self._n_generations += 1
            self.temperature = np.maximum(self.temperature*self.rt, 1e-200)
        return self._collect(fitness, y)
