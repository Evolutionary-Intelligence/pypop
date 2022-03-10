import numpy as np

from optimizers.es.es import ES
from optimizers.es.maes import MAES


class LMMAES(ES):
    """Limited-Memory Matrix Adaptation Evolution Strategy (LMMAES).

    Reference
    ---------
    Loshchilov, I., Glasmachers, T. and Beyer, H.G., 2019.
    Large scale black-box optimization by limited-memory matrix adaptation.
    IEEE Transactions on Evolutionary Computation, 23(2), pp.353-358.
    https://ieeexplore.ieee.org/abstract/document/8410043
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        n_evolution_paths = 4 + int(3 * np.log(self.ndim_problem))
        self.n_evolution_paths = options.get('n_evolution_paths', n_evolution_paths)  # m in Algorithm 1
        self.c_s = options.get('c_s', self._set_c_s())  # c_sigma in Algorithm 1
        self._s_1 = self._set__s_1()  # for Line 13 in Algorithm 1
        self._s_2 = self._set__s_2()  # for Line 13 in Algorithm 1
        self._c_d = self._set__c_d()
        self._c_c = self._set__c_c()

    def _set_c_s(self):
        return 2 * self.n_individuals / self.ndim_problem

    def _set__s_1(self):
        return 1 - self.c_s

    def _set__s_2(self):
        return np.sqrt(self._mu_eff * self.c_s * (2 - self.c_s))

    def _set__c_d(self):
        return 1 / (self.ndim_problem * np.power(1.5, np.arange(self.n_evolution_paths)))

    def _set__c_c(self):
        return self.n_individuals / (self.ndim_problem * np.power(4.0, np.arange(self.n_evolution_paths)))

    def initialize(self, is_restart=False):
        self.c_s = self._set_c_s()  # c_sigma in Algorithm 1
        self._s_1 = self._set__s_1()
        self._s_2 = self._set__s_2()  # for Line 13 in Algorithm 1
        self._c_c = self._set__c_c()
        z = np.empty((self.n_individuals, self.ndim_problem))  # Gaussian noise for mutation
        d = np.empty((self.n_individuals, self.ndim_problem))  # search directions
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        s = np.zeros((self.ndim_problem,))  # evolution path (p in Algorithm 1)
        tm = np.zeros((self.n_evolution_paths, self.ndim_problem))  # transformation matrix M
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return z, d, mean, s, tm, y

    def iterate(self, z=None, d=None, mean=None, tm=None, y=None, args=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return z, d, y
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            d[k] = z[k]
            for j in range(np.minimum(self._n_generations, self.n_evolution_paths)):
                d[k] = (1 - self._c_d[j]) * d[k] + self._c_d[j] * tm[j] * np.dot(tm[j], d[k])
            y[k] = self._evaluate_fitness(mean + self.sigma * d[k], args)
        return z, d, y

    def _update_distribution(self, z=None, d=None, mean=None, s=None, tm=None, y=None):
        order = np.argsort(y)
        d_w, z_w = np.zeros((self.ndim_problem,)), np.zeros((self.ndim_problem,))
        for k in range(self.n_parents):
            d_w += self._w[k] * d[order[k]]
            z_w += self._w[k] * z[order[k]]
        # update distribution mean
        mean += (self.sigma * d_w)
        # update evolution path (p_c, s) and low-rank transformation matrix (tm)
        s = self._s_1 * s + self._s_2 * z_w
        for k in range(self.n_evolution_paths):  # rank-m
            tm[k] = (1 - self._c_c[k]) * tm[k] + np.sqrt(self._mu_eff * self._c_c[k] * (2 - self._c_c[k])) * z_w
        # update global step-size
        self.sigma *= np.exp(self.c_s / 2 * (np.sum(np.power(s, 2)) / self.ndim_problem - 1))
        return mean, s, tm

    def restart_initialize(self, z=None, d=None, mean=None, s=None, tm=None, y=None):
        is_restart = ES.restart_initialize(self)
        if is_restart:
            z, d, mean, s, tm, y = self.initialize(is_restart)
        return z, d, mean, s, tm, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        return MAES.optimize(self, fitness_function, args)
