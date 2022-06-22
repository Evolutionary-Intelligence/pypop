import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class HJDSM(Optimizer):
    """Hooke-Jeeves Direct Search Method(DSM)
    Reference
    --------------
    Hooke, R. and Jeeves, T.A., 1961.
    “Direct search” solution of numerical and statistical problems.
    Journal of the ACM, 8(2), pp.212-229.
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self.sigma = options.get('sigma')
        self.alpha = options.get('alpha')
        self.beta = options.get('beta')
        self._n_generations = 0
        self.x = options.get('x')
        self.e_matrix = np.identity(self.ndim_problem)

    def initialize(self, is_restart=False):
        x = self._initialize_x(is_restart)
        x_new = x.copy()
        return x, x_new

    def iterate(self, x, x_new):
        for i in range(self.ndim_problem):
            if self._evaluate_fitness(x_new + self.sigma * self.e_matrix[i]) \
                    < self._evaluate_fitness(x_new):
                x_new += self.sigma * self.e_matrix[i]
            elif self._evaluate_fitness(x_new - self.sigma * self.e_matrix[i]) \
                    < self._evaluate_fitness(x_new):
                x_new -= self.sigma * self.e_matrix[i]
        return x, x_new

    def _update_distribution(self, x, x_new):
        if self._evaluate_fitness(x_new) < self._evaluate_fitness(x):
            x, x_new = x_new, x_new + self.alpha * (x_new - x)
        else:
            self.sigma, x_new = self.sigma * self.beta, x
        return x, x_new

    def optimize(self, fitness_function=None):
        fitness = Optimizer.optimize(self, fitness_function)
        x, x_new = self.initialize()
        while True:
            x, x_new = self.iterate(x, x_new)
            y = self._evaluate_fitness(x)
            if self.record_fitness:
                fitness.append(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            x, x_new = self._update_distribution(x, x_new)
        results = self._collect_results(fitness)
        return results

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose_frequency):
            best_so_far_y = -self.best_so_far_y if self._is_maximization else self.best_so_far_y
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, best_so_far_y, np.min(y), self.n_function_evaluations))

    def _initialize_x(self, is_restart=False):
        if is_restart or (self.x is None):
            x = self.rng_initialization.uniform(self.initial_lower_boundary,
                                                   self.initial_upper_boundary)
        else:
            x = np.copy(self.x)
        return x

    def _collect_results(self, fitness, mean=None):
        results = Optimizer._collect_results(self, fitness)
        results['sigma'] = self.sigma
        results['_n_generations'] = self._n_generations
        return results
