import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class CEM(Optimizer):
    """Cross-Entropy Method(CE)
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
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:
            self.n_individuals = 100
        if self.n_parents is None:
            self.n_parents = 10
        self._n_generations = 0
        self.sigma = np.ones((self.ndim_problem,)) * options.get('sigma')
        self.mean = options.get('mean')  # mean of Gaussian search distribution
        if self.mean is None:  # 'mean' has priority over 'x'
            self.mean = options.get('x')

    def initialize(self, is_restart=False):
        mean = self._initialize_mean(is_restart)
        y = np.empty((self.n_individuals,))
        x = np.empty((self.n_individuals, self.ndim_problem))
        return mean, x, y

    def iterate(self, mean, x, y):
        for i in range(self.n_individuals):
            x[i] = mean + np.dot(self.sigma, self.rng_optimization.standard_normal((self.ndim_problem,)))
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
        fitness = Optimizer.optimize(self, fitness_function)
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

    def _initialize_mean(self, is_restart=False):
        if is_restart or (self.mean is None):
            mean = self.rng_initialization.uniform(self.initial_lower_boundary,
                                                   self.initial_upper_boundary)
        else:
            mean = np.copy(self.mean)
        return mean

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose_frequency):
            best_so_far_y = -self.best_so_far_y if self._is_maximization else self.best_so_far_y
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect_results(self, fitness, mean=None):
        results = Optimizer._collect_results(self, fitness)
        results['sigma'] = self.sigma
        results['_n_generations'] = self._n_generations
        return results
