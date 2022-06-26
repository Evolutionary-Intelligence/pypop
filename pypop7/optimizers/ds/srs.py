import numpy as np

from pypop7.optimizers.dsm.dsm import DSM


class SRS(DSM):
    """Simple Random Search(SRS)
    Reference
    --------------
    M. T. Rosenstein and A. G. Barto
    Robot Weightlifting By Direct Policy Search
    International Joint Conference on Artificial Intelligence, 2001
    """
    def __init__(self, problem, options):
        DSM.__init__(self, problem, options)
        self.x = options.get('x')
        self.gamma = options.get('gamma')
        self.sigma_min = options.get('sigma_min')

    def initialize(self, is_restart=False):
        x = self._initialize_x(is_restart)
        x_best = x.copy()
        y_best = self._evaluate_fitness(x_best)
        return x, x_best, y_best

    def iterate(self, x, x_best, y_best):
        delta_x = self.rng_optimization.normal(0, self.sigma, 1)
        y = self._evaluate_fitness(x + delta_x)
        if y < y_best:
            x_best = x + delta_x
            y_best = y
        prob = np.random.random()
        if prob < self.beta:
            x += self.alpha * delta_x
        else:
            x += self.alpha * (x_best - x)
        return x, x_best, y, y_best

    def optimize(self, fitness_function=None):
        fitness = DSM.optimize(self, fitness_function)
        x, x_best, y_best = self.initialize()
        while True:
            self.sigma = max(self.gamma * self.sigma, self.sigma_min)
            x, x_best, y, y_best = self.iterate(x, x_best, y_best)
            if self.record_fitness:
                fitness.append(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results
