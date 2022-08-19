import numpy as np
import random

from pypop7.optimizers.ep.ep import EP


class FEP(EP):
    """Fast Evolutionary Programming(FEP)
        Reference
        -----------
        X. Yao, Y, Liu, G. Lin
        Evolutionary Programming Made Faster
        IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 3, NO. 2, JULY 1999
        https://ieeexplore.ieee.org/abstract/document/771163
    """
    def __init__(self, problem, options):
        EP.__init__(self, problem, options)
        self.t = 1.0 / np.sqrt(2 * np.sqrt(self.ndim_problem))
        self.t1 = 1.0 / np.sqrt(2 * self.ndim_problem)
        self.q = options.get('q', 10)
        self.init_n = options.get('init_n', 3)

    def initialize(self, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        n = self.init_n * np.ones((self.n_individuals, self.ndim_problem))
        for i in range(self.n_individuals):
            x[i] = self._initialize_mean()
            y[i] = self._evaluate_fitness(x[i])
        return x, y, n

    def iterate(self, x, y, n):
        new_x = np.empty((self.n_individuals, self.ndim_problem))
        new_n = np.empty((self.n_individuals, self.ndim_problem))
        new_y = np.empty((self.n_individuals,))
        rand1 = self.rng_optimization.standard_normal(self.ndim_problem)  # gaussian normal
        rand2 = self.rng_optimization.standard_cauchy(self.ndim_problem)  # cauchy normal
        for i in range(self.n_individuals):
            for j in range(self.ndim_problem):
                new_x[i][j] = x[i][j] + n[i][j] * rand2[j]
                new_n[i][j] = n[i][j] * np.exp(self.t1 * self.rng_optimization.standard_normal(1) + self.t * rand1[j])
            new_x[i] = np.clip(new_x[i], self.lower_boundary, self.upper_boundary)
            new_y[i] = self._evaluate_fitness(new_x[i])
        new_x = np.vstack((new_x, x))
        new_y = np.hstack((new_y, y))
        new_n = np.vstack((new_x, x))
        win = np.zeros((2 * self.n_individuals,))
        list1 = list(range(0, 2 * self.n_individuals))
        for i in range(2 * self.n_individuals):
            list2 = random.sample(list1, self.q)
            for j in list2:
                if new_y[i] >= new_y[j]:
                    win[i] += 1
        order = np.argsort(win)
        for i in range(self.n_individuals):
            x[i] = new_x[order[i]]
            y[i] = new_y[order[i]]
            n[i] = new_n[order[i]]
        return x, y, n

    def optimize(self, fitness_function=None):
        fitness = EP.optimize(self, fitness_function)
        x, y, n = self.initialize()
        fitness.extend(y)
        while True:
            x, y, n = self.iterate(x, y, n)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results
