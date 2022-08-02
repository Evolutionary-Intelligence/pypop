import numpy as np

from pypop7.optimizers.ds.ds import DS


class CS(DS):
    """Coordinate Search (CS).

    AKA: alternating directions, alternating variable search, axial relaxation, local variation, compass search

    NOTE that the current implementation is a highly simplified version of the Coordinate Search algorithm,
        since its original version needs (3 ** n - 1) samples for each iteration in the worst case, where n
        is the dimensionality of the problem. Such a worst-case complexity limits its applicability for
        large-scale optimization scenarios. Instead, here we use the opportunistic strategy for simplicity.
        See Algorithm 3 from [Torczon, 1997, SIAM-JO] for details.

    Reference
    ---------
    Torczon, V., 1997.
    On the convergence of pattern search algorithms.
    SIAM Journal on Optimization, 7(1), pp.1-25.
    https://epubs.siam.org/doi/abs/10.1137/S1052623493250780

    Fermi, E. and Metropolis N., 1952.
    Numerical solution of a minimum problem.
    Los Alamos Scientific Lab., Los Alamos, NM.
    https://www.osti.gov/servlets/purl/4377177
    """
    def __init__(self, problem, options):
        DS.__init__(self, problem, options)
        self.gamma = options.get('gamma', 0.5)  # decreasing factor of step-size

    def initialize(self, args=None, is_restart=False):
        x = self._initialize_x(is_restart)  # initial point
        y = self._evaluate_fitness(x, args)  # fitness
        return x, y

    def iterate(self, args=None, x=None, fitness=None):
        improved = False
        for i in range(self.ndim_problem):
            for sgn in [-1, 1]:
                if self._check_terminations():
                    return x
                xx = np.copy(x)
                xx[i] += sgn * self.sigma
                y = self._evaluate_fitness(xx, args)
                if self.record_fitness:
                    fitness.append(y)
                if y < self.best_so_far_y:
                    x = xx  # greedy / opportunistic
                    improved = True
                    break
        if not improved:
            self.sigma *= self.gamma  # alpha
        return x

    def restart_initialize(self, args=None, x=None, y=None, fitness=None):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self.n_restart += 1
            self.sigma = np.copy(self._sigma_bak)
            x, y = self.initialize(args, is_restart)
            fitness.append(y)
            self._fitness_list = [self.best_so_far_y]
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x, y = self.initialize(args)
        fitness.append(y)
        while True:
            x = self.iterate(args, x, fitness)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                x, y = self.restart_initialize(args, x, y, fitness)
        results = self._collect_results(fitness)
        return results
