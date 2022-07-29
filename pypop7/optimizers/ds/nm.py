import numpy as np

from pypop7.optimizers.ds.ds import DS


class NM(DS):
    """Nelder-Mead simplex method (NM).

    Reference
    ---------
    Singer, S. and Nelder, J., 2009.
    Nelder-mead algorithm.
    Scholarpedia, 4(7), p.2928.
    http://var.scholarpedia.org/article/Nelder-Mead_algorithm

    Wright, M.H., 1996.
    Direct search methods: Once scorned, now respectable.
    Pitman Research Notes in Mathematics Series, pp.191-208.
    https://nyuscholars.nyu.edu/en/publications/direct-search-methods-once-scorned-now-respectable

    Nelder, J.A. and Mead, R., 1965.
    A simplex method for function minimization.
    The Computer Journal, 7(4), pp.308-313.
    https://academic.oup.com/comjnl/article-abstract/7/4/308/354237
    """
    def __init__(self, problem, options):
        DS.__init__(self, problem, options)
        self.alpha = options.get('alpha', 1)  # reflection factor
        self.gamma = options.get('gamma', 2)  # expansion factor
        self.beta = options.get('beta', 0.5)  # contraction factor
        self.shrinkage = options.get('', 0.5)  # shrinkage factor
        self.n_individuals = self.ndim_problem + 1

    def initialize(self, args=None, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))  # simplex
        y = np.empty((self.n_individuals,))  # fitness
        x[0] = self._initialize_x(is_restart)  # as suggested in [Wright, 1996]
        y[0] = self._evaluate_fitness(x[0], args)
        for i in range(1, self.n_individuals):
            if self._check_terminations():
                return x, y
            x[i] = x[0]
            x[i, i - 1] += self.sigma * self.rng_initialization.uniform(-1, 1)
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y

    def iterate(self, x=None, y=None, args=None):
        order = np.argsort(y)
        l, h = order[0], order[-1]  # index of lowest and highest points
        p_mean = np.mean(x[order[:-1]], axis=0)  # centroid of all vertices except the worst
        p_star = (1 + self.alpha) * p_mean - self.alpha * x[h]  # reflection
        y_star = self._evaluate_fitness(p_star, args)
        if y_star < y[l]:
            p_star_star = self.gamma * p_star + (1 - self.gamma) * p_mean  # expansion
            y_star_star = self._evaluate_fitness(p_star_star, args)
            if y_star_star < y_star:  # as suggested in [Wright, 1996]
                x[h], y[h] = p_star_star, y_star_star
            else:
                x[h], y[h] = p_star, y_star
        else:
            if np.all(y_star > y[order[:-1]]):
                if y_star <= y[h]:
                    x[h], y[h] = p_star, y_star
                p_star_star = self.beta * x[h] + (1 - self.beta) * p_mean
                y_star_star = self._evaluate_fitness(p_star_star, args)
                if y_star_star > y[h]:
                    for i in range(1, self.n_individuals):  # shrinkage
                        x[order[i]] = x[l] + self.shrinkage * (x[order[i]] - x[l])
                        y[order[i]] = self._evaluate_fitness(x[order[i]], args)
                else:
                    x[h], y[h] = p_star_star, y_star_star
            else:
                x[h], y[h] = p_star, y_star
        return x, y

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
            fitness.extend(y)
            self._fitness_list = [self.best_so_far_y]
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x, y = self.initialize(args)
        fitness.extend(y)
        while True:
            x, y = self.iterate(x, y, args)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                x, y = self.restart_initialize(args, x, y, fitness)
        results = self._collect_results(fitness)
        return results
