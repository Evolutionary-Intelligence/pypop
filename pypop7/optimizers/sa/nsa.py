import numpy as np

from pypop7.optimizers.sa.sa import RS
from pypop7.optimizers.sa.sa import SA


class NSA(SA):
    """Noisy Simulated Annealing (NSA).

    .. note:: This is a *slightly modified* version of `NSA` for continuous (rather discrete) optimization
       **with noisy observations**.

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
              and with the following particular setting (`key`):
                * 'x' - initial (starting) point (`array_like`).

    Attributes
    ----------
    x     : `array_like`
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
        self.is_noisy = options.get('is_noisy', True)
        self.schedule = options.get('schedule', 'linear')  # schedule for sampling intensity
        assert self.schedule in ['linear', 'quadratic'],\
            'Currently only two (*linear* or *quadratic*) schedules are supported for sampling intensity.'
        self.n_samples = options.get('n_samples')
        self.rc = options.get('cr', 0.99)  # reducing factor of temperature
        self._tk = 0

    def initialize(self, args=None):
        if self.x is None:  # starting point
            x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        else:
            x = np.copy(self.x)
        y = self._evaluate_fitness(x, args)
        self.parent_x, self.parent_y = np.copy(x), np.copy(y)
        return y

    def iterate(self, args=None):
        x = self.parent_x + self.sigma*self.rng_optimization.standard_normal((self.ndim_problem,))
        self._tk += self.rng_optimization.exponential()
        if self.schedule == 'linear':
            n_tk = self._tk
        else:  # quadratic
            n_tk = np.power(self._tk, 2)
        if self.n_samples is None:
            n_samples = self.rng_optimization.poisson(n_tk) + 1
        else:
            n_samples = self.n_samples
        y, parent_y = [], []
        for _ in range(n_samples):
            if self._check_terminations():
                break
            y.append(self._evaluate_fitness(x, args))
            self._n_generations += 1
            self._print_verbose_info(y)
        if self.is_noisy:  # for noisy optimization
            for _ in range(n_samples):
                if self._check_terminations():
                    break
                parent_y.append(self._evaluate_fitness(self.parent_x, args))
                self._n_generations += 1
                self._print_verbose_info(parent_y)
        else:  # for static optimization
            parent_y = self.parent_y
        diff = np.mean(parent_y) - np.mean(y)
        if (diff >= 0) or (self.rng_optimization.random() < np.exp(diff / self.temperature)):
            self.parent_x, self.parent_y = np.copy(x), np.copy(y)
        if not self.is_noisy:  # for static optimization
            parent_y = []
        return y, parent_y

    def optimize(self, fitness_function=None, args=None):
        super(RS, self).optimize(fitness_function)
        fitness = [self.initialize(args)]  # to store all fitness generated during search
        self._print_verbose_info(fitness[0])
        while not self._check_terminations():
            y, parent_y = self.iterate(args)
            if self.saving_fitness:
                fitness.extend(y)
                fitness.extend(parent_y)
            self.temperature *= self.rc  # temperature reducing
        return self._collect_results(fitness)
