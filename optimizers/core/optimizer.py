import time
from enum import IntEnum
import numpy as np


# helper class
class Terminations(IntEnum):
    NO_TERMINATION = 0
    MAX_FUNCTION_EVALUATIONS = 1
    MAX_RUNTIME = 2
    FITNESS_THRESHOLD = 3


class Optimizer(object):
    """Base class of all optimizers for continuous black-box minimization.
    """
    def __init__(self, problem, options):
        # problem-related settings
        self.problem = problem
        self._is_maximization = problem.get('_is_maximization', False)
        self.fitness_function = problem.get('fitness_function')
        self.ndim_problem = problem['ndim_problem']
        self.upper_boundary = problem.get('upper_boundary')
        self.lower_boundary = problem.get('lower_boundary')
        self.initial_upper_boundary = problem.get('initial_upper_boundary', self.upper_boundary)
        self.initial_lower_boundary = problem.get('initial_lower_boundary', self.lower_boundary)
        self.problem_name = problem.get('problem_name')
        if (self.problem_name is None) and hasattr(self.fitness_function, '__name__'):
            self.problem_name = self.fitness_function.__name__

        # optimizer-related options
        self.max_function_evaluations = options.get('max_function_evaluations', np.Inf)
        self.max_runtime = options.get('max_runtime', np.Inf)
        self.fitness_threshold = options.get('fitness_threshold', -np.Inf)
        if self._is_maximization:
            self.fitness_threshold *= -1
        self.n_individuals = options.get('n_individuals')
        self.seed_rng = options.get('seed_rng')
        if self.seed_rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(self.seed_rng)
        self.seed_initialization = options.get('seed_initialization', self.rng.integers(np.iinfo(np.int64).max))
        self.rng_initialization = np.random.default_rng(self.seed_initialization)
        self.seed_optimization = options.get('seed_optimization', self.rng.integers(np.iinfo(np.int64).max))
        self.rng_optimization = np.random.default_rng(self.seed_optimization)
        self.record_fitness = options.get('record_fitness', False)
        self.record_fitness_frequency = options.get('record_fitness_frequency', 1000)
        self.verbose = options.get('verbose', True)
        self.verbose_frequency = options.get('verbose_frequency', 10)

        # auxiliary members
        self.Terminations = Terminations
        self.n_function_evaluations = options.get('n_function_evaluations', 0)
        self.start_function_evaluations = None
        self.time_function_evaluations = options.get('time_function_evaluations', 0)
        self.runtime = options.get('runtime', 0)
        self.start_time = None
        self.best_so_far_y = options.get('best_so_far_y', np.Inf)
        if self._is_maximization:
            self.best_so_far_y *= -1
        self.best_so_far_x = None
        self.termination_signal = None
        self.fitness = None

    def _evaluate_fitness(self, x, args=None):
        self.start_function_evaluations = time.time()
        if args is None:
            y = self.fitness_function(x)
        else:
            y = self.fitness_function(x, args=args)
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += 1
        # update best-so-far solution and fitness
        if (not self._is_maximization) and (y < self.best_so_far_y):
            self.best_so_far_x, self.best_so_far_y = np.copy(x), y
        if self._is_maximization and (-y > self.best_so_far_y):
            self.best_so_far_x, self.best_so_far_y = np.copy(x), -y
        return float(y)

    def _check_terminations(self):
        self.runtime = time.time() - self.start_time
        if self.n_function_evaluations >= self.max_function_evaluations:
            termination_signal = True, Terminations.MAX_FUNCTION_EVALUATIONS
        elif self.runtime >= self.max_runtime:
            termination_signal = True, Terminations.MAX_RUNTIME
        elif not self._is_maximization and (self.best_so_far_y <= self.fitness_threshold):
            termination_signal = True, Terminations.FITNESS_THRESHOLD
        elif self._is_maximization and (self.best_so_far_y >= self.fitness_threshold):
            termination_signal = True, Terminations.FITNESS_THRESHOLD
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
        # use 1-based index
        index = np.arange(1, len(fitness), self.record_fitness_frequency)
        # recover 0-based index via - 1
        index = np.append(index, len(fitness)) - 1
        self.fitness = np.stack((index, fitness[index]), 1)

    def _collect_results(self):
        if self._is_maximization:
            self.best_so_far_y *= -1
        return {'best_so_far_x': self.best_so_far_x,
                'best_so_far_y': self.best_so_far_y,
                'n_function_evaluations': self.n_function_evaluations,
                'runtime': time.time() - self.start_time,
                'termination_signal': self.termination_signal,
                'time_function_evaluations': self.time_function_evaluations,
                'fitness': self.fitness}

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def optimize(self, fitness_function=None):
        raise NotImplementedError
