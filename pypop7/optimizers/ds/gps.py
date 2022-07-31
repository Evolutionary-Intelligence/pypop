import numpy as np

from pypop7.optimizers.ds.ds import DS


class GPS(DS):
    """Generalized Pattern Search (GPS).

    NOTE that "to converge to a local minimum, certain conditions must be met. The set of directions must
        be a positive spanning set, which means that we can construct any point using a nonnegative
        linear combination of the directions. A positive spanning set ensures that at least one of the
        directions is a descent direction from a location with a nonzero gradient."
        (from [Kochenderfer&Wheeler, 2019])

    Reference
    ---------
    Kochenderfer, M.J. and Wheeler, T.A., 2019.
    Algorithms for optimization.
    MIT Press.
    https://algorithmsbook.com/optimization/files/chapter-7.pdf
    (See Algorithm 7.6 (Page 106) for details.)

    Regis, R.G., 2016.
    On the properties of positive spanning sets and positive bases.
    Optimization and Engineering, 17(1), pp.229-262.
    https://link.springer.com/article/10.1007/s11081-015-9286-x

    Torczon, V., 1997.
    On the convergence of pattern search algorithms.
    SIAM Journal on Optimization, 7(1), pp.1-25.
    https://epubs.siam.org/doi/abs/10.1137/S1052623493250780
    """
    def __init__(self, problem, options):
        DS.__init__(self, problem, options)
        self.gamma = options.get('gamma', 0.5)  # decreasing factor of step-size (γ)

    def initialize(self, args=None, is_restart=False):
        x = self._initialize_x(is_restart)  # initial point
        y = self._evaluate_fitness(x, args)  # fitness
        # random directions
        d = self.rng_initialization.standard_normal(size=(self.ndim_problem + 1, self.ndim_problem))
        i_d = [i for i in range(d.shape[0])]  # index of used directions
        return x, y, d, i_d

    def iterate(self, args=None, x=None, d=None, i_d=None, fitness=None):
        improved, best_so_far_y = False, self.best_so_far_y
        for i in range(d.shape[0]):
            if self._check_terminations():
                return i_d
            x = self.best_so_far_x + self.sigma * d[i_d[i]]  # opportunistic
            y = self._evaluate_fitness(x, args)
            if self.record_fitness:
                fitness.append(y)
            if y < best_so_far_y:
                improved = True
                i_d = [i_d[i]] + i_d[:i] + i_d[(i + 1):]  # dynamic ordering
                break
        if not improved:
            self.sigma *= self.gamma  # alpha
        return i_d

    def restart_initialize(self, args=None, x=None, y=None, d=None, i_d=None, fitness=None):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self.n_restart += 1
            self.sigma = np.copy(self._sigma_bak)
            x, y, d, i_d = self.initialize(args, is_restart)
            fitness.append(y)
            self._fitness_list = [self.best_so_far_y]
        return x, y, d, i_d

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x, y, d, i_d = self.initialize(args)
        fitness.append(y)
        while True:
            i_d = self.iterate(args, x, d, i_d, fitness)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                x, y, d, i_d = self.restart_initialize(args, x, y, d, i_d, fitness)
        results = self._collect_results(fitness)
        return results
