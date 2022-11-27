import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class CC(Optimizer):
    """Cooperative Coevolution(CC)

    This is the **base** (abstract) class for all CC classes.

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
              and with the following particular setting (`key`):
                * 'x' - initial (starting) point (`array_like`).

    Attributes
    ----------
    x     : `array_like`
            initial (starting) point.
    Methods
    -------

    Reference
    ---------
    F. Gomez, J. Schmidhuber, R. Miikkulainen
    Accelerated Neural Evolution through Cooperatively Coevolved Synapses
    https://jmlr.org/papers/v9/gomez08a.html
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # number of offspring, offspring population size (λ: lambda)
            self.n_individuals = 4 + int(3 * np.log(self.ndim_problem))  # for small population setting
        if self.n_parents is None:  # number of parents, parental population size (μ: mu)
            self.n_parents = int(self.n_individuals / 2)
        self._n_generations = 0
        self.x = options.get('x')

    def initialize(self, is_restart=False):
        raise NotImplementedError

    def iterate(self, mean, x, y):
        raise NotImplementedError

    def crossover(self, x, y, cross_type):
        s1 = x.copy()
        s2 = y.copy()
        if cross_type == 'one_point':
            place = np.random.randint(0, self.ndim_problem)
            for i in range(place):
                s1[i] = y[i]
                s2[i] = x[i]
        elif cross_type == 'two_point':
            place1 = np.random.randint(0, self.ndim_problem)
            place2 = np.random.randint(place1, self.ndim_problem)
            for i in range(place1, place2):
                s1[i] = y[i]
                s2[i] = x[i]
        elif cross_type == 'uniform':
            for i in range(self.ndim_problem):
                rand = np.random.random()
                if rand < 0.5:
                    s1[i] = y[i]
                    s2[i] = x[i]
        return [s1, s2]

    def mutate(self, x):
        for i in range(self.ndim_problem):
            rand = np.random.random()
            if rand < self.prob_mutate:
                x[i] = self.lower_boundary[i] + np.random.random() *\
                       (self.upper_boundary[i] - self.lower_boundary[i])
        return x

    def _initialize_x(self, is_restart=False):
        if is_restart or (self.x is None):
            x = self.rng_initialization.uniform(self.initial_lower_boundary,
                                                self.initial_upper_boundary)
        else:
            x = np.copy(self.x)
        return x

    def _collect_results(self, fitness, mean=None):
        results = Optimizer._collect_results(self, fitness)
        results['_n_generations'] = self._n_generations
        return results

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))
