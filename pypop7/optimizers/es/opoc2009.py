import numpy as np

from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.opoc import OPOC


class OPOC2009(OPOC):
    """(1+1)-Cholesky-CMA-ES (OPOC).

    Reference
    ---------
    Suttorp, T., Hansen, N. and Igel, C., 2009.
    Efficient covariance matrix update for variable metric evolution strategies.
    Machine Learning, 75(2), pp.167-197.
    https://link.springer.com/article/10.1007/s10994-009-5102-1
    (See Algorithm 2 for details.)
    """
    def __init__(self, problem, options):
        OPOC.__init__(self, problem, options)

    def initialize(self, args=None, is_restart=False):
        mean, y, a, best_so_far_y, p_s = OPOC.initialize(self, args, is_restart)
        p_c = np.zeros((self.ndim_problem,))  # evolution path
        a_i = np.diag(np.ones((self.ndim_problem,)))  # inverse of Cholesky factors
        return mean, y, a, a_i, best_so_far_y, p_s, p_c

    def iterate(self, args=None, mean=None, a=None, a_i=None, best_so_far_y=None, p_s=None, p_c=None):
        # sample and evaluate (only one) offspring
        z = self.rng_optimization.standard_normal((self.ndim_problem,))
        x = mean + self.sigma * np.dot(a, z)
        y = self._evaluate_fitness(x, args)
        if y <= best_so_far_y:
            l_s = 1
        else:
            l_s = 0
        p_s = (1 - self.c_p) * p_s + self.c_p * l_s
        self.sigma *= np.exp(self.eta_sigma * (p_s - self.p_ts) / (1 - self.p_ts))
        if y <= best_so_far_y:
            mean, best_so_far_y = x, y
            if p_s < self.p_t:
                p_c = (1 - self.c_c) * p_c + np.sqrt(self.c_c * (2 - self.c_c)) * np.dot(a, z)
                alpha = 1 - self.c_cov
            else:
                p_c *= 1 - self.c_c
                alpha = 1 - self.c_cov + self.c_cov * self.c_c * (2 - self.c_c)
            beta = self.c_cov
            w = np.dot(a_i, p_c)
            w_norm = np.power(np.linalg.norm(w), 2)
            s_w_norm = np.sqrt(1 + beta / alpha * w_norm)
            a = np.sqrt(alpha) * a + np.sqrt(alpha) / w_norm * (s_w_norm - 1) * np.dot(
                p_c[:, np.newaxis], w[np.newaxis, :])
            a_i = 1 / np.sqrt(alpha) * a_i - 1 / (np.sqrt(alpha) * w_norm) * (1 - 1 / s_w_norm) * np.dot(
                w[:, np.newaxis], np.dot(w[np.newaxis, :], a_i))
        return mean, y, a, a_i, best_so_far_y, p_s, p_c

    def restart_initialize(self, args=None, mean=None, y=None, a=None, a_i=None,
                           best_so_far_y=None, p_s=None, p_c=None, fitness=None):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self.n_restart += 1
            self.sigma = np.copy(self._sigma_bak)
            mean, y, a, a_i, best_so_far_y, p_s, p_c = self.initialize(args, is_restart)
            fitness.append(y)
            self._fitness_list = [best_so_far_y]
        return mean, y, a, a_i, best_so_far_y, p_s, p_c

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        mean, y, a, a_i, best_so_far_y, p_s, p_c = self.initialize(args)
        fitness.append(y)
        while True:
            mean, y, a, a_i, best_so_far_y, p_s, p_c = self.iterate(
                args, mean, a, a_i, best_so_far_y, p_s, p_c)
            if self.record_fitness:
                fitness.append(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                mean, y, a, a_i, best_so_far_y, p_s, p_c = self.restart_initialize(
                    args, mean, y, a, a_i, best_so_far_y, p_s, p_c, fitness)
        results = self._collect_results(fitness, mean)
        return results
