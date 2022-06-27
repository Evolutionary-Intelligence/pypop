import numpy as np

from pypop7.optimizers.es.opoc2009 import OPOC2009


class OPOA(OPOC2009):
    """(1+1)-Active-CMA-ES (OPOA).

    Reference
    ---------
    Arnold, D.V. and Hansen, N., 2010, July.
    Active covariance matrix adaptation for the (1+1)-CMA-ES.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 385-392). ACM.
    https://dl.acm.org/doi/abs/10.1145/1830483.1830556
    """
    def __init__(self, problem, options):
        OPOC2009.__init__(self, problem, options)
        self.c_m = options.get('c_m', 0.4 / (np.power(self.ndim_problem, 1.6) + 1))
        self.k = options.get('k', 5)
        self._ancestors = []

    def initialize(self, args=None, is_restart=False):
        mean, y, a, a_i, best_so_far_y, p_s, p_c = OPOC2009.initialize(self, args, is_restart)
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
            self._ancestors.append(y)
            mean, best_so_far_y = x, y
            if p_s < self.p_t:
                p_c = (1 - self.c_c) * p_c + np.sqrt(self.c_c * (2 - self.c_c)) * np.dot(a, z)
                alpha = 1 - self.c_cov
            else:
                p_c *= 1 - self.c_c
                alpha = 1 - self.c_cov + self.c_cov * self.c_c * (2 - self.c_c)
            w = np.dot(a_i, p_c)
            w_power = np.dot(w, w)
            alpha = np.sqrt(alpha)
            beta = alpha / w_power * (np.sqrt(1 + self.c_cov / (1 - self.c_cov) * w_power) - 1)
            a = alpha * a + beta * np.dot(p_c[:, np.newaxis], w[np.newaxis, :])
            a_i = 1 / alpha * a_i - beta / (np.power(alpha, 2) + alpha * beta * w_power) * np.dot(
                w[:, np.newaxis], np.dot(w[np.newaxis, :], a_i))
        if len(self._ancestors) >= self.k and y > self._ancestors[-self.k]:
            del self._ancestors[0]
            z_power = np.dot(z, z)
            if 1 < self.c_m * (2 * z_power - 1):
                c_m = 1 / (2 * z_power - 1)
            else:
                c_m = self.c_m
            alpha = np.sqrt(1 + c_m)
            beta = alpha / z_power * (np.sqrt(1 - self.c_m / (1 - self.c_m) * z_power) - 1)
            a = alpha * a + beta * np.dot(np.dot(a, z[:, np.newaxis]), z[np.newaxis, :])
            a_i = 1 / alpha * a_i - beta / (np.power(alpha, 2) + alpha * beta * z_power) * np.dot(
                z[:, np.newaxis], np.dot(z[np.newaxis, :], a_i))
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
            self._ancestors = []
        return mean, y, a, a_i, best_so_far_y, p_s, p_c
