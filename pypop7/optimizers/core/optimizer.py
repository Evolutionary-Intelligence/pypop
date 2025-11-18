"""For simplicity, only the *minimization* form of the optimization problem
    is considered in this `core` module, since *maximization* could be
    easily transformed into *minimization*.
"""
import time
# only for the class `Terminations`
from enum import IntEnum

import numpy as np

from pypop7.utils.constants import I64M


class Terminations(IntEnum):
    """Helper class used by all global optimizers.

    A total of 4 termination conditions are considered:
    1. MAX_FUNCTION_EVALUATIONS: When the maximum of function evaluations
       is reached, the optimizer will stop to run immediately.
    2. MAX_RUNTIME: When the maximum of actual runtime is reached, the
       optimizer will stop to run immediately.
    3. FITNESS_THRESHOLD: When the threshold of the *best-so-far* fitness
       is reached, the optimizer will stop to run immediately.
    4. EARLY_STOPPING: When the condition of early stopping is reached,
       the optimizer will stop to run immediately.

    To make any of 4 termination conditions really working, the
    corresponding setting (e.g., threshold) needs to be predefined during
    the *initialization* phase of the optimizer.
    """
    NO_TERMINATION = 0
    # maximum of function evaluations
    MAX_FUNCTION_EVALUATIONS = 1
    # maximal runtime to be allowed
    MAX_RUNTIME = 2
    # when *best-so-far* fitness is below the predefined threshold
    FITNESS_THRESHOLD = 3
    # when *best-so-far* fitness does not improve for a long time
    EARLY_STOPPING = 4


class Optimizer(object):
    """Base (aka **abstract**) class of both individual-based and population-based
        **approximately global** optimizers for continuous black-box **minimization**.

    Parameters
    ----------
    problem : dict
              problem arguments with the following common settings (`keys`):
                * 'fitness_function'       - objective function to be **minimized** (`func`),
                * 'ndim_problem'           - number of dimensionality (`int`),
                * 'upper_boundary'         - upper boundary of search range (`array_like`),
                * 'lower_boundary'         - lower boundary of search range (`array_like`),
                * 'initial_upper_boundary' - upper boundary only for initialization (`array_like`),
                * 'initial_lower_boundary' - lower boundary only for initialization (`array_like`),
                * 'problem_name'           - problem name (`str`).
    options : dict
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`).
    """
    def __init__(self, problem, options):
        # **problem-related** basic settings #
        # to be *minimized* approximately globally
        self.fitness_function = problem.get('fitness_function')
        self.ndim_problem = problem.get('ndim_problem')  # number of problem dimensionality
        assert self.ndim_problem > 0
        # mainly for both the *initialization* process and the *constrained sampling* process:
        # constrained sampling is mainly covered in its extension library called `PyCoPop7`
        # upper boundary of search range
        self.upper_boundary = problem.get('upper_boundary')
        # lower boundary of search range
        self.lower_boundary = problem.get('lower_boundary')
        # only for the *initialization* process
        self.initial_upper_boundary = problem.get('initial_upper_boundary',
                                                  self.upper_boundary)
        self.initial_lower_boundary = problem.get('initial_lower_boundary',
                                                  self.lower_boundary)
        # mainly for the *printing* purpose:
        # should be of data type `str`
        self.problem_name = problem.get('problem_name')
        if (self.problem_name is None) and hasattr(self.fitness_function, '__name__'):
            self.problem_name = self.fitness_function.__name__

        # *optimizer-related* basic options #
        # NOT use `deep copy` in order to save CPU memory
        self.options = options
        # for `MAX_FUNCTION_EVALUATIONS` in `Terminations`
        self.max_function_evaluations = options.get('max_function_evaluations', np.inf)
        # for `MAX_RUNTIME` in `Terminations`
        self.max_runtime = options.get('max_runtime', np.inf)
        # for `FITNESS_THRESHOLD` in `Terminations`
        self.fitness_threshold = options.get('fitness_threshold', -np.inf)
        # mainly for *population-based randomized optimizers*
        # *offspring* population size
        self.n_individuals = options.get('n_individuals')
        # *parent* population size
        self.n_parents = options.get('n_parents')
        # for strict randomness control to ensure ***repeatability*** AMAP
        self.seed_rng = options.get('seed_rng')
        if self.seed_rng is None:  # highly recommended to explicitly set `seed_rng`
            self.rng = np.random.default_rng()  # NOT use it, if possible
        else:
            self.rng = np.random.default_rng(self.seed_rng)
        # better to set both `seed_initialization` and `seed_optimization`
        #   automatically according to `seed_rng`
        self.seed_initialization = options.get('seed_initialization',
                                               self.rng.integers(np.iinfo(np.int64).max))
        self.rng_initialization = np.random.default_rng(self.seed_initialization)
        self.seed_optimization = options.get('seed_optimization',
                                             self.rng.integers(np.iinfo(np.int64).max))
        self.rng_optimization = np.random.default_rng(self.seed_optimization)

        # basic options for saving results and printing progress
        # frequency of saving fitness when outputing results, to save memory space
        self.saving_fitness = options.get('saving_fitness', 0)
        # only for printing in console
        self.verbose = options.get('verbose', 10)

        # auxiliary members mainly for *count* and *summary*
        self.Terminations, self.termination_signal = Terminations, 0  # 0 -> `NO_TERMINATION`
        self.n_function_evaluations = options.get('n_function_evaluations', 0)
        self.start_function_evaluations = None
        self.time_function_evaluations = options.get('time_function_evaluations', 0)
        self.runtime, self.start_time = options.get('runtime', 0), None
        self.best_so_far_y, self.best_so_far_x = options.get('best_so_far_y', np.inf), None
        self.fitness = None
        self.is_restart = options.get('is_restart', True)
        # all members of *early stopping* (closed by default according to following settings)
        self.early_stopping_evaluations = options.get('early_stopping_evaluations', np.inf)
        self.early_stopping_threshold = options.get('early_stopping_threshold', 0.0)
        self._counter_early_stopping, self._base_early_stopping = 0, self.best_so_far_y

    def _evaluate_fitness(self, x, args=None):
        self.start_function_evaluations = time.time()
        if args is None:
            y = self.fitness_function(x)
        else:
            y = self.fitness_function(x, args=args)
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += 1
        # update best-so-far solution (x) and fitness (y)
        if y < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x), y
        # update all settings related to early stopping
        if (self._base_early_stopping - y) <= self.early_stopping_threshold:
            self._counter_early_stopping += 1
        else:
            self._counter_early_stopping, self._base_early_stopping = 0, y
        return float(y)

    def _check_terminations(self):
        self.runtime = time.time() - self.start_time
        if self.n_function_evaluations >= self.max_function_evaluations:
            termination_signal = True, Terminations.MAX_FUNCTION_EVALUATIONS
        elif self.runtime >= self.max_runtime:
            termination_signal = True, Terminations.MAX_RUNTIME
        elif self.best_so_far_y <= self.fitness_threshold:
            termination_signal = True, Terminations.FITNESS_THRESHOLD
        elif self._counter_early_stopping >= self.early_stopping_evaluations:
            termination_signal = True, Terminations.EARLY_STOPPING
        else:
            termination_signal = False, Terminations.NO_TERMINATION
        self.termination_signal = termination_signal[1]
        return termination_signal[0]

    def _compress_fitness(self, fitness):
        fitness = np.array(fitness)
        # arrange in non-increasing order
        for i in range(len(fitness) - 1):
            if fitness[i] < fitness[i + 1]:
                fitness[i + 1] = fitness[i]
        if self.saving_fitness == 1:
            self.fitness = np.stack((np.arange(1, len(fitness) + 1), fitness), 1)
        elif self.saving_fitness > 1:
            # use 1-based index
            index = np.arange(1, len(fitness), self.saving_fitness)
            # recover 0-based index via - 1
            index = np.append(index, len(fitness)) - 1
            self.fitness = np.stack((index, fitness[index]), 1)
            # recover 1-based index
            self.fitness[0, 0], self.fitness[-1, 0] = 1, len(fitness)

    def _check_success(self):
        if (self.upper_boundary is not None) and (self.lower_boundary is not None) and (
                np.any(self.lower_boundary > self.best_so_far_x) or np.any(self.best_so_far_x > self.upper_boundary)):
            return False
        elif np.isnan(self.best_so_far_y) or np.any(np.isnan(self.best_so_far_x)):
            return False
        return True

    def _collect(self, fitness):
        if self.saving_fitness:
            self._compress_fitness(fitness[:self.n_function_evaluations])
        return {'best_so_far_x': self.best_so_far_x,
                'best_so_far_y': self.best_so_far_y,
                'n_function_evaluations': self.n_function_evaluations,
                'runtime': time.time() - self.start_time,
                'termination_signal': self.termination_signal,
                'time_function_evaluations': self.time_function_evaluations,
                'fitness': self.fitness,
                'success': self._check_success()}

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def optimize(self, fitness_function=None):
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        fitness = []  # to store all fitness generated during evolution/optimization
        return fitness
