import time

from optimizers.core.optimizer import Optimizer


class RS(Optimizer):
    """Random (Stochastic) Search (RS).

    The individual-based iteration (generation) process is used here.

    Reference
    ---------
    Brooks, S.H., 1958.
    A discussion of random methods for seeking maxima.
    Operations Research, 6(2), pp.244-251.
    https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        verbose_options = options.get('verbose_options')
        if (verbose_options is None) or (verbose_options.get('frequency_verbose') is None):
            self.verbose_options['frequency_verbose'] = 1000
        self.x = options.get('x')  # starting search point

    def initialize(self):
        pass

    def iterate(self):  # for each iteration (generation)
        pass

    def optimize(self, fitness_function=None, args=None):  # for all iterations (generations)
        self.start_time = time.time()
        fitness = []  # store all fitness generated during search
        if fitness_function is not None:
            self.fitness_function = fitness_function
        is_initialization = True
        while not self._check_terminations():
            if is_initialization:
                x = self.initialize()
                is_initialization = False
            else:
                x = self.iterate()
            y = self._evaluate_fitness(x, args)
            if self.record_options['record_fitness']:
                fitness.append(y)
            self._print_verbose_info()
        if self.record_options['record_fitness']:
            self._compress_fitness(fitness[:self.n_function_evaluations])
        return self._collect_results()

    def _print_verbose_info(self):
        if self.verbose_options['verbose']:
            if not self.n_function_evaluations % self.verbose_options['frequency_verbose']:
                info = '  * Evaluations {:d}: best_so_far_y {:7.5e}'
                print(info.format(self.n_function_evaluations, self.best_so_far_y))
