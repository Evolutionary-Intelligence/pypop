import numpy as np

from pypop7.optimizers.es.es import ES


class OPOC(ES):
    """(1+1)-Cholesky-CMA-ES (OPOC).

    Reference
    ---------
    Igel, C., Suttorp, T. and Hansen, N., 2006, July.
    A computational efficient covariance matrix update and a (1+1)-CMA for evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 453-460). ACM.
    https://dl.acm.org/doi/abs/10.1145/1143997.1144082
    (See Algorithm 2 for details.)
    """
    def __init__(self, problem, options):
        options['n_individuals'] = 1
        options['n_parents'] = 1
        ES.__init__(self, problem, options)
        if self.eta_sigma is None:
            self.eta_sigma = 1 / (1 + self.ndim_problem / 2)
        self.p_ts = options.get('p_ts', 2 / 11)
        self.c_p = options.get('c_p', 1 / 12)
        self.c_c = options.get('c_c', 2 / (self.ndim_problem + 2))
        self.c_cov = options.get('c_cov', 2 / (np.power(self.ndim_problem, 2) + 6))
        self.p_t = options.get('p_t', 0.44)
        self.c_a = options.get('c_a', np.sqrt(1 - self.c_cov))

    def initialize(self, args=None, is_restart=False):
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        y = self._evaluate_fitness(mean, args)  # fitness
        a = np.diag(np.ones(self.ndim_problem,))  # linear transformation
        best_so_far_y, p_s, l_s = np.copy(y), self.p_ts, 0
        return mean, y, a, best_so_far_y, p_s, l_s

    def iterate(self, args=None, mean=None, a=None, best_so_far_y=None, p_s=None, l_s=None):
        # sample and evaluate (only one) offspring
        z = self.rng_optimization.standard_normal((self.ndim_problem,))
        x = mean + self.sigma * np.dot(a, z)
        y = self._evaluate_fitness(x, args)
        if y <= best_so_far_y:
            mean, best_so_far_y, l_s = x, y, 1
            if p_s < self.p_t:
                z_norm, c_a = np.power(np.linalg.norm(z), 2), np.power(self.c_a, 2)
                a = self.c_a * a + self.c_a / z_norm * (np.sqrt(1 + ((1 - c_a) * z_norm) / c_a) - 1) * np.dot(
                    np.dot(a, z[:, np.newaxis]), z[np.newaxis, :])
        else:
            l_s = 0
        p_s = (1 - self.c_p) * p_s + self.c_p * l_s
        self.sigma *= np.exp(self.eta_sigma * (p_s - self.p_ts / (1 - self.p_ts) * (1 - p_s)))
        return mean, y, a, best_so_far_y, p_s, l_s

    def restart_initialize(self, args=None, mean=None, y=None,
                           a=None, best_so_far_y=None, p_s=None, l_s=None, fitness=None):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self.n_restart += 1
            self.sigma = np.copy(self._sigma_bak)
            mean, y, a, best_so_far_y, p_s, l_s = self.initialize(args, is_restart)
            fitness.append(y)
            self._fitness_list = [best_so_far_y]
        return mean, y, a, best_so_far_y, p_s, l_s

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        mean, y, a, best_so_far_y, p_s, l_s = self.initialize(args)
        fitness.append(y)
        while True:
            mean, y, a, best_so_far_y, p_s, l_s = self.iterate(args, mean, a, best_so_far_y, p_s, l_s)
            if self.record_fitness:
                fitness.append(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                mean, y, a, best_so_far_y, p_s, l_s = self.restart_initialize(
                    args, mean, y, a, best_so_far_y, p_s, l_s, fitness)
        results = self._collect_results(fitness, mean)
        return results
