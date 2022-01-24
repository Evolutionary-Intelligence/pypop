import numpy as np

from optimizers.es.es import ES


class RES(ES):
    """Rechenberg's Evolution Strategy with 1/5th success rule (RES).

    Reference
    ---------
    Hansen, N., Arnold, D.V. and Auger, A., 2015.
    Evolution strategies.
    In Springer Handbook of Computational Intelligence (pp. 871-898). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007%2F978-3-662-43505-2_44
    """
    def __init__(self, problem, options):
        options['n_individuals'] = 1  # mandatory setting for RES
        ES.__init__(self, problem, options)
        if self.eta_sigma is None:
            self.eta_sigma = 1 / np.sqrt(self.ndim_problem + 1)

    def initialize(self, args=None, is_restart=None):
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        y = self._evaluate_fitness(mean, args)  # fitness
        best_so_far_y = np.copy(y)
        return mean, y, best_so_far_y

    def iterate(self, args=None, mean=None):
        x = mean + self.sigma * self.rng_optimization.standard_normal((self.ndim_problem,))
        y = self._evaluate_fitness(x, args)
        return x, y

    def restart_initialize(self, args=None, mean=None, y=None, best_so_far_y=None, fitness=None):
        is_restart = ES.restart_initialize(self)
        if is_restart:
            mean, y, best_so_far_y = self.initialize(args, is_restart)
            fitness.append(y)
        return mean, y, best_so_far_y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        ES.optimize(self, fitness_function)
        mean, y, best_so_far_y = self.initialize(args)
        fitness = [y]  # store all fitness generated during evolution
        while True:
            # sample and evaluate (only one) offspring
            x, y = self.iterate(args, mean)
            if self.record_fitness:
                fitness.append(y)
            if self._check_terminations():
                break
            self.sigma *= np.power(np.exp(float(y < best_so_far_y) - 1 / 5), self.eta_sigma)
            self._n_generations += 1
            self._print_verbose_info(y)
            if y < best_so_far_y:
                mean, best_so_far_y = x, y
            mean, y, best_so_far_y = self.restart_initialize(args, mean, y, best_so_far_y, fitness)
        results = self._collect_results(fitness)
        results['mean'] = mean
        return results
