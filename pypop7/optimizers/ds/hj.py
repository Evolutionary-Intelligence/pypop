import numpy as np

from pypop7.optimizers.ds.ds import DS


class HJ(DS):
    """Hooke-Jeeves direct search method (HJ).

    Reference
    ---------
    Kochenderfer, M.J. and Wheeler, T.A., 2019.
    Algorithms for optimization.
    MIT Press.
    https://algorithmsbook.com/optimization/files/chapter-7.pdf
    (See Algorithm 7.5 (Page 104) for details.)

    Kaupe Jr, A.F., 1963.
    Algorithm 178: Direct search.
    Communications of the ACM, 6(6), pp.313-314.
    https://dl.acm.org/doi/pdf/10.1145/366604.366632

    Hooke, R. and Jeeves, T.A., 1961.
    “Direct search” solution of numerical and statistical problems.
    Journal of the ACM, 8(2), pp.212-229.
    https://dl.acm.org/doi/10.1145/321062.321069
    """
    def __init__(self, problem, options):
        DS.__init__(self, problem, options)
        self.gamma = options.get('gamma', 0.5)  # decreasing factor of step-size (γ)

    def initialize(self, args=None, is_restart=False):
        x = self._initialize_x(is_restart)  # initial point
        y = self._evaluate_fitness(x, args)  # fitness
        best_so_far_x, best_so_far_y = np.copy(x), y  # base point
        return x, y, best_so_far_x, best_so_far_y

    def iterate(self, args=None, x=None, best_so_far_x=None, best_so_far_y=None, fitness=None):
        improved = False
        for i in range(self.ndim_problem):
            for sgn in [-1, 1]:
                if self._check_terminations():
                    return best_so_far_x, best_so_far_y
                xx = np.copy(x)
                xx[i] += sgn * self.sigma
                y = self._evaluate_fitness(xx, args)
                if self.record_fitness:
                    fitness.append(y)
                if y < best_so_far_y:
                    best_so_far_x, best_so_far_y, improved = np.copy(xx), y, True
        if not improved:
            self.sigma *= self.gamma  # alpha
        return best_so_far_x, best_so_far_y

    def restart_initialize(self, args=None, x=None, y=None, best_so_far_x=None, best_so_far_y=None, fitness=None):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self.n_restart += 1
            self.sigma = np.copy(self._sigma_bak)
            x, y, best_so_far_x, best_so_far_y = self.initialize(args, is_restart)
            fitness.append(y)
            self._fitness_list = [best_so_far_y]
        return x, y, best_so_far_x, best_so_far_y

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x, y, best_so_far_x, best_so_far_y = self.initialize(args)
        fitness.append(y)
        while True:
            x, y = self.iterate(args, x, best_so_far_x, best_so_far_y, fitness)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                x, y, best_so_far_x, best_so_far_y = self.restart_initialize(
                    args, x, y, best_so_far_x, best_so_far_y, fitness)
        results = self._collect_results(fitness)
        return results
