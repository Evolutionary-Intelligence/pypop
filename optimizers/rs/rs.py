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
        self.x = options.get('x')  # starting search point
        if options.get('verbose_frequency') is None:
            self.verbose_frequency = 1000  # reset the default value from 10 to 1000

    def initialize(self):
        raise NotImplementedError

    def iterate(self):  # for each iteration (generation)
        raise NotImplementedError

    def _print_verbose_info(self):
        if self.verbose and (not self.n_function_evaluations % self.verbose_frequency):
            info = '  * Evaluations {:d}: best_so_far_y {:7.5e}'
            print(info.format(self.n_function_evaluations, self.best_so_far_y))

    def optimize(self, fitness_function=None, args=None):  # for all iterations (generations)
        Optimizer.optimize(self, fitness_function)
        fitness = []  # store all fitness generated during search
        is_initialization = True
        while not self._check_terminations():
            if is_initialization:
                x = self.initialize()
                is_initialization = False
            else:
                x = self.iterate()
            y = self._evaluate_fitness(x, args)
            if self.record_fitness:
                fitness.append(y)
            self._print_verbose_info()
        return self._collect_results(fitness)
