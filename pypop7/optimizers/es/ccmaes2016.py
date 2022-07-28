import numpy as np
from scipy.linalg import solve_triangular as st

from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.opoa2015 import cholesky_update


class CCMAES2016(ES):
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.c_s = options.get('c_s', self._set_c_s())
        self.d = options.get('d', self._set_d())
        self.c_c = options.get('c_c', self._set_c_c())
        self.c_1 = options.get('c_1', self._set_c_1())
        self.c_mu = options.get('c_mu', self._set_c_mu())

    def _set_c_s(self):
        return self._mu_eff / (self.ndim_problem + self._mu_eff)

    def _set_d(self):
        return 1 + np.sqrt(self._mu_eff / self.ndim_problem)

    def _set_c_c(self):
        return (4 + self._mu_eff / self.ndim_problem) / (
                self.ndim_problem + 4 + 2 * self._mu_eff / self.ndim_problem)

    def _set_c_1(self):
        return 2 / (np.power(self.ndim_problem, 2) + self._mu_eff)

    def _set_c_mu(self):
        return self._mu_eff / (np.power(self.ndim_problem, 2) + self._mu_eff)

    def initialize(self, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        a = np.diag(np.ones((self.ndim_problem,)))  # cholesky factor
        p_s = np.zeros((self.ndim_problem,))  # evolution path for CSA
        p_c = np.zeros((self.ndim_problem,))  # evolution path for CMA
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        if is_restart:
            self.c_s = self._set_c_s()
            self.d = self._set_d()
            self.c_c = self._set_c_c()
            self.c_1 = self._set_c_1()
            self.c_mu = self._set_c_mu()
        return x, mean, a, p_s, p_c, y

    def iterate(self, x=None, mean=None, a=None, y=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            x[k] = mean + self.sigma * np.dot(a, self.rng_optimization.standard_normal((self.ndim_problem,)))
            y[k] = self._evaluate_fitness(x[k])
        return x, y

    def _update_distribution(self, x=None, mean=None, a=None, p_s=None, p_c=None, y=None):
        order = np.argsort(y)[:self.n_parents]
        mean_bak = np.dot(self._w, x[order])
        mean_diff = (mean_bak - mean) / self.sigma
        p_c = (1 - self.c_c) * p_c + np.sqrt(self.c_c * (2 - self.c_c) * self._mu_eff) * mean_diff
        p_s = (1 - self.c_s) * p_s + np.sqrt(self.c_s * (2 - self.c_s) * self._mu_eff) * st(a, mean_diff, lower=True)
        a *= np.sqrt(1 - self.c_1 - self.c_mu)
        a = cholesky_update(a, np.sqrt(self.c_1) * p_c, False)
        for i in range(self.n_parents):
            a = cholesky_update(a, np.sqrt(self.c_mu * self._w[i]) * (x[order[i]] - mean) / self.sigma, False)
        self.sigma *= np.exp(self.c_s / self.d * (np.sqrt(np.dot(p_s, p_s)) / self._e_chi - 1))
        mean = mean_bak
        return mean, a, p_s, p_c

    def restart_initialize(self, x=None, mean=None, a=None, p_s=None, p_c=None, y=None):
        is_restart = ES.restart_initialize(self)
        if is_restart:
            x, mean, a, p_s, p_c, y = self.initialize(is_restart)
        return x, mean, a, p_s, p_c, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, a, p_s, p_c, y = self.initialize()
        while True:
            x, y = self.iterate(x, mean, a, y)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, a, p_s, p_c = self._update_distribution(x, mean, a, p_s, p_c, y)
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                x, mean, a, p_s, p_c, y = self.restart_initialize(x, mean, a, p_s, p_c, y)
        results = self._collect_results(fitness, mean)
        results['p_s'] = p_s
        results['p_c'] = p_c
        return results
