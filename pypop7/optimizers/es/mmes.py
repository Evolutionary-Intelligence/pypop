import numpy as np
from scipy.stats import norm

from pypop7.optimizers.es.es import ES


class MMES(ES):
    """Mixture Model-based Evolution Strategy (MMES).
    Reference
    ---------
    He, X., Zheng, Z. and Zhou, Y., 2021.
    MMES: Mixture model-based evolution strategy for large-scale optimization.
    IEEE Transactions on Evolutionary Computation, 25(2), pp.320-333.
    https://ieeexplore.ieee.org/abstract/document/9244595
    See the official Matlab version from He:
    https://github.com/hxyokokok/MMES
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        # number of candidate direction vectors
        self.m = options.get('m', 2 * int(np.ceil(np.sqrt(self.ndim_problem))))
        # learning rate of evolution path
        self.c_c = options.get('c_c', 0.4 / np.sqrt(self.ndim_problem))
        self.ms = options.get('ms', 4)  # mixing strength (l)
        # for paired test adaptation (PTA)
        self.c_s = options.get('c_s', 0.3)  # learning rate of step-size adaptation
        self.a_z = options.get('a_z', 0.05)  # target significance level
        # minimal distance of updating evolution paths (T)
        self.distance = options.get('distance', np.ceil(1 / self.c_c))
        # success probability of geometric distribution (different from 4/n in the original paper)
        self.c_a = options.get('c_a', 3.8 / self.ndim_problem)  # same as Matlab code
        self.gamma = options.get('gamma', 1 - np.power(1 - self.c_a, self.m))
        self._n_mirror_sampling = None
        self._z_1 = np.sqrt(1 - self.gamma)
        self._z_2 = np.sqrt(self.gamma / self.ms)
        self._p_1 = 1 - self.c_c
        self._p_2 = np.sqrt(self.c_c * (2 - self.c_c))
        self._w_1 = 1 - self.c_s
        self._w_2 = np.sqrt(self.c_s * (2 - self.c_s))

    def initialize(self, args=None, is_restart=False):
        self._n_mirror_sampling = int(np.ceil(self.n_individuals / 2))
        x = np.zeros((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        p = np.zeros((self.ndim_problem,))  # evolution path (Line 2 in Algorithm 1)
        w = 0  # Line 3
        q = np.zeros((self.m, self.ndim_problem))  # candidate direction vectors (Line 4)
        t = np.zeros((self.m,))  # Line 5 (generations recorded)
        v = np.arange(self.m)  # Line 6 (indexes to evolution paths)
        y = np.tile(self._evaluate_fitness(mean, args), (self.n_individuals,))  # fitness
        return x, mean, p, w, q, t, v, y

    def iterate(self, x=None, mean=None, q=None, v=None, y=None, args=None):
        for k in range(self._n_mirror_sampling):  # mirror sampling
            zq = np.zeros((self.ndim_problem,))
            for _ in range(self.ms):
                j_k = v[(self.m - self.rng_optimization.geometric(self.c_a) % self.m) - 1]
                zq += self.rng_optimization.standard_normal() * q[j_k]
            z = self._z_1 * self.rng_optimization.standard_normal((self.ndim_problem,))
            z += self._z_2 * zq  # Line 13
            x[k] = mean + self.sigma * z  # Line 14
            if (self._n_mirror_sampling + k) < self.n_individuals:
                x[self._n_mirror_sampling + k] = mean - self.sigma * z
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _update_distribution(self, x, mean, p, w, q, t, v, y, y_bak):
        order = np.argsort(y)
        y.sort()  # Line 16
        mean_w = np.zeros((self.ndim_problem,))
        for k in range(self.n_parents):  # Line 17
            mean_w += self._w[k] * x[order[k]]
        p = self._p_1 * p + self._p_2 * np.sqrt(self._mu_eff) * (mean_w - mean) / self.sigma  # Line 18
        mean = mean_w  # for Line 17
        if self._n_generations < self.m:
            q[self._n_generations] = p
        else:
            k_star = np.argmin(t[v[1:]] - t[v[:(self.m - 1)]])  # Line 19
            k_star += 1
            if t[v[k_star]] - t[v[k_star - 1]] > self.distance:  # Line 20
                k_star = 0  # Line 21
            v = np.append(np.append(v[:k_star], v[(k_star + 1):]), v[k_star])
            t[v[-1]], q[v[-1]] = self._n_generations, p  # Line 25, 26
        # for success-based adaptation of mutation strength
        l_w = np.dot(self._w, y_bak[:self.n_parents] > y[:self.n_parents])  # Line 27
        w = self._w_1 * w + self._w_2 * np.sqrt(self._mu_eff) * (2 * l_w - 1)  # Line 28
        self.sigma *= np.exp(norm.cdf(w) - 1 + self.a_z)  # Line 29
        return mean, p, w, q, t, v

    def restart_initialize(self, args=None, x=None, mean=None, p=None, w=None, q=None,
                           t=None, v=None, y=None, fitness=None):
        is_restart = ES.restart_initialize(self)
        if is_restart:
            x, mean, p, w, q, t, v, y = self.initialize(args, is_restart)
            fitness.append(y[0])
        return x, mean, p, w, q, t, v, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p, w, q, t, v, y = self.initialize(args)
        fitness.append(y[0])
        while True:
            y_bak = np.copy(y)
            x, y = self.iterate(x, mean, q, v, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, p, w, q, t, v = self._update_distribution(x, mean, p, w, q, t, v, y, y_bak)
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                x, mean, p, w, q, t, v, y = self.restart_initialize(
                    args, x, mean, p, w, q, t, v, y, fitness)
        results = self._collect_results(fitness, mean)
        results['p'] = p
        results['w'] = w
        return results
