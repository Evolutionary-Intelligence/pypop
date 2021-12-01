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

    def initialize(self, args=None):
        mean = self._initialize_mean()  # mean of Gaussian search distribution
        y = self._evaluate_fitness(mean, args)  # fitness
        return y

    def iterate(self, args=None):
        noise = self.rng_optimization.standard_normal((self.ndim_problem,))
        y = self._evaluate_fitness(self.best_so_far_x + self.sigma * noise, args)
        return y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        ES.optimize(self, fitness_function)
        fitness = [self.initialize()]  # store all fitness generated during evolution
        while True:
            y_bak = np.copy(self.best_so_far_y)
            # sample and evaluate (only one) offspring
            y = self.iterate(args)
            if self.record_fitness:
                fitness.append(y)
            if self._check_terminations():
                break
            self.sigma *= np.power(np.exp(float(y < y_bak) - 1 / 5), self.eta_sigma)
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results
