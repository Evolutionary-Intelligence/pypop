import time
import numpy as np

from optimizers.rs.rs import RS


class PRS(RS):
    """Pure Random Search (PRS).

    Reference
    ---------
    Brooks, S.H., 1958.
    A discussion of random methods for seeking maxima.
    Operations Research, 6(2), pp.244-251.
    https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244
    """
    def __init__(self, problem, options):
        RS.__init__(self, problem, options)
        self.sampling_distribution = options.get('sampling_distribution')
        if self.sampling_distribution is None:
            self.sampling_distribution = 1  # default: 1 -> uniformly distributed
        if self.sampling_distribution not in [0, 1]:  # 0 -> normally distributed
            raise ValueError('Only support uniformly or normally distributed random sampling.')

    def _sample(self, rng):
        if self.sampling_distribution == 0:
            x = rng.standard_normal(size=(self.ndim_problem,))
        else:
            x = rng.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        return x

    def initialize(self):
        if self.x is None:
            x = self._sample(self.rng_initialization)
        else:
            x = np.copy(self.x)
        return x

    def iterate(self):
        # draw sample (individual)
        return self._sample(self.rng_optimization)

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
                x = self.iterate()  # sample (individual)
            # evaluate fitness
            self.start_function_evaluations = time.time()
            y = self.fitness_function(x)
            self.time_function_evaluations += time.time() - self.start_function_evaluations
            self.n_function_evaluations += 1
            # update best-so-far solution and fitness
            if y < self.best_so_far_y:
                self.best_so_far_y = y
                self.best_so_far_x = np.copy(x)
            if self.record_options['record_fitness']:
                fitness.append(float(y))
            self._print_verbose_info()
            if self._check_terminations():
                break
        if self.record_options['record_fitness']:
            self._compress_fitness(fitness[:self.n_function_evaluations])
        return self._collect_results()
