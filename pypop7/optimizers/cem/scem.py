import numpy as np

from pypop7.optimizers.cem.cem import CEM


class SCEM(CEM):
    """Cross-Entropy Method(CEM)
    Reference
    -----------
    P. T. de Boer, D. P. Kroese, S. Mannor, R. Y. Rubinstein, (2003)
    A Tutorial on the Cross-Entropy Method
    http://web.mit.edu/6.454/www/www_fall_2003/gew/CEtutorial.pdf
    T. Hoem-de-Mello, R. Y. Rubinstein,
    Estimation of Rare Event Probabilities using Cross-Entropy
    Proceedings of the 2002 Winter Simulation Conference
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1172900
    The official python version:
    https://pypi.org/project/cross-entropy-method/
    """
    def __init__(self, problem, options):
        CEM.__init__(self, problem, options)

    def initialize(self, is_restart=False):
        mean = self._initialize_mean(is_restart)
        y = np.empty((self.n_individuals,))
        x = np.empty((self.n_individuals, self.ndim_problem))
        return mean, x, y

    def iterate(self, mean, x, y):
        for i in range(self.n_individuals):
            x[i] = self.rng_optimization.normal(mean, self.sigma, (1, self.ndim_problem))
            y[i] = self._evaluate_fitness(x[i])
        return x, y

    def _update_parameters(self, mean, x, y):
        order = np.argsort(y)
        new_x = np.empty((self.n_parents, self.ndim_problem))
        for j in range(self.n_parents):
            new_x[j] = x[order[j]]
        mean = np.mean(new_x, axis=0)
        self.sigma = np.std(new_x, axis=0)
        return mean

    def optimize(self, fitness_function=None, args=None):
        fitness = CEM.optimize(self, fitness_function)
        mean, x, y = self.initialize()
        while True:
            x, y = self.iterate(mean, x, y)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            mean = self._update_parameters(mean, x, y)
        results = self._collect_results(fitness, mean)
        return results
