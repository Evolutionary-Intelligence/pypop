import time
import numpy as np

from optimizers.rs.rs import RS


class RHC(RS):
    """Random (Stochastic) Hill Climber (RHC).

    Reference
    ---------
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/hillclimber.py
    """
    def __init__(self, problem, options):
        RS.__init__(self, problem, options)
        self.initial_std = options.get('initial_std', 1.0)
        self.global_std = options.get('global_std', 0.1)

    def initialize(self):
        if self.x is None:
            x = self.rng_initialization.standard_normal(size=(self.ndim_problem,))
            x *= self.initial_std
        else:
            x = np.copy(self.x)
        return x

    def iterate(self):
        mutation = self.rng_optimization.standard_normal(size=(self.ndim_problem,))
        return self.best_so_far_x + self.global_std * mutation

    def optimize(self, fitness_function=None):
        self.start_time = time.time()
        fitness = []  # store all fitness generated during search
        if fitness_function is not None:
            self.fitness_function = fitness_function
        is_initialization = True
        while True:
            if is_initialization:
                x = self.initialize()
                is_initialization = False
            else:
                x = self.iterate()  # mutate the best-so-far individual
            y = self._evaluate_fitness(x)
            if self.record_options['record_fitness']:
                fitness.append(y)
            self._print_verbose_info()
            if self._check_terminations():
                break
        if self.record_options['record_fitness']:
            self._compress_fitness(fitness[:self.n_function_evaluations])
        return self._collect_results()
