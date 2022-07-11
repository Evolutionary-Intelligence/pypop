import numpy as np

from pypop7.optimizers.es.es import ES


class LMCMAES(ES):
    """Limited Memory Covariance Matrix Adaptation Evolution Strategy (LMCMAES, (μ/μ_w,λ)-LM-CMA-ES).

    Reference
    ---------
    Loshchilov, I., 2014, July.
    A computationally efficient limited memory CMA-ES for large scale optimization.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 397-404). ACM.
    https://dl.acm.org/doi/abs/10.1145/2576768.2598294
    (See Algorithm 6 for details.)

    See the official C++ version from Loshchilov:
    https://sites.google.com/site/lmcmaeses/
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.c_s = options.get('c_s', 0.3)  # learning rate for population success rule (PSR)
        self.d_s = options.get('d_s', 1)
        self.m = options.get('m', self.n_individuals)  # number of direction vectors
        self.n_steps = options.get('n_steps', self.m)  # target number of generations between vectors
        self.c_c = options.get('c_c', 1 / self.m)  # learning rate for evolution path
        # learning rate for covariance matrix adaptation
        self.c_1 = options.get('c_1', 1 / (10 * np.log(self.ndim_problem + 1)))
        self.z_star = options.get('z_star', 0.25)  # target success rate for PSR
        self._a = np.sqrt(1 - self.c_1)  # for Algorithm 3
        self._c = 1 / np.sqrt(1 - self.c_1)  # for Algorithm 4
        self._bd_1 = np.sqrt(1 - self.c_1)  # for Line 13 and 14
        self._bd_2 = self.c_1 / (1 - self.c_1)  # for Line 13 and 14
        self._p_c_1 = 1 - self.c_c  # for Line 9
        self._p_c_2 = None  # for Line 9
        self._rr = None  # for PSR
        self._j = None
        self._l = None

    def initialize(self, is_restart=None):
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        p_c = np.zeros((self.ndim_problem,))  # evolution path
        s = 0  # for PSR of global step-size adaptation
        vm = np.empty((self.m, self.ndim_problem))
        pm = np.empty((self.m, self.ndim_problem))
        b = np.empty((self.m,))
        d = np.empty((self.m,))
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        self._p_c_2 = np.sqrt(self.c_c * (2 - self.c_c) * self._mu_eff)
        self._rr = np.arange(self.n_individuals * 2, 0, -1) - 1
        self._j = [None] * self.m
        self._l = [None] * self.m
        return mean, x, p_c, s, vm, pm, b, d, y

    def _a_z(self, z=None, pm=None, vm=None, b=None):  # Cholesky factor - vector update
        # Algorithm 3 Az()
        x = np.copy(z)
        for t in range(np.minimum(self.m, self._n_generations)):
            x = self._a * x + b[self._j[t]] * np.dot(vm[self._j[t]], z) * pm[self._j[t]]
        return x

    def iterate(self, mean=None, x=None, pm=None, vm=None, y=None, b=None, args=None):
        for k in range(self.n_individuals):  # Line 4
            if self._check_terminations():
                return x, y
            z = self.rng_optimization.standard_normal((self.ndim_problem,))  # Line 5
            x[k] = mean + self.sigma * self._a_z(z, pm, vm, b)  # Line 6
            y[k] = self._evaluate_fitness(x[k], args)  # Line 7
        return x, y

    def _a_inv_z(self, v=None, vm=None, d=None, i=None):  # inverse Cholesky factor - vector update
        # Algorithm 4 Ainvz()
        x = np.copy(v)
        for t in range(0, i):
            x = self._c * x - d[self._j[t]] * np.dot(vm[self._j[t]], x) * vm[self._j[t]]
        return x

    def _update_distribution(self, mean=None, x=None, p_c=None, s=None, vm=None, pm=None,
                             b=None, d=None, y=None, y_bak=None):
        mean_bak = np.dot(self._w, x[np.argsort(y)[:self.n_parents]])  # Line 8
        p_c = self._p_c_1 * p_c + self._p_c_2 * (mean_bak - mean) / self.sigma  # Line 9
        if self._n_generations < self.m:  # before self.m generations
            self._j[self._n_generations] = self._n_generations
            i_min = self._n_generations
        else:
            i_min = 1  # start from the latter (1) of pairwise evolution paths
            d_min = self._l[self._j[i_min]] - self._l[self._j[i_min - 1]]
            for j in range(2, self.m):
                d_cur = self._l[self._j[j]] - self._l[self._j[j - 1]]
                if d_cur < d_min:
                    d_min, i_min = d_cur, j
            # if all pairwise distances exceed self.n_steps, start from 0
            i_min = 0 if d_min >= self.n_steps else i_min
            # update indexes of evolution paths (self._j[i_min] is index of evolution path needed to delete)
            updated = self._j[i_min]
            for j in range(i_min, self.m - 1):
                self._j[j] = self._j[j + 1]
            self._j[self.m - 1] = updated
        # note that the latest evolution path is stored in self._j[np.minimum(self.m - 1, self._n_generations)]
        pm[self._j[np.minimum(self.m - 1, self._n_generations)]] = p_c  # add the latest evolution path
        self._l[self._j[np.minimum(self.m - 1, self._n_generations)]] = self._n_generations  # update its generation
        # since self._j[i_min] is deleted, all vectors (from vm) depending on it need to be computed again
        for i in range(i_min, np.minimum(self.m, self._n_generations + 1)):
            vm[self._j[i]] = self._a_inv_z(pm[self._j[i]], vm, d, i)
            v_n = np.dot(vm[self._j[i]], vm[self._j[i]])
            bd_3 = np.sqrt(1 + self._bd_2 * v_n)
            b[self._j[i]] = self._bd_1 / v_n * (bd_3 - 1)
            d[self._j[i]] = 1 / (self._bd_1 * v_n) * (1 - 1 / bd_3)
        y.sort()
        if self._n_generations > 0:
            r = np.argsort(np.hstack((y, y_bak)))  # for Line 15
            z_psr = np.sum(self._rr[r < self.n_individuals] - self._rr[r >= self.n_individuals])  # Line 15
            z_psr = z_psr / np.power(self.n_individuals, 2) - self.z_star  # Line 15
            s = (1 - self.c_s) * s + self.c_s * z_psr  # Line 17
            self.sigma *= np.exp(s / self.d_s)  # Line 18
        return mean_bak, p_c, s, vm, pm, b, d

    def restart_initialize(self, args=None, mean=None, x=None, p_c=None, s=None, vm=None, pm=None,
                           b=None, d=None, y=None):
        is_restart = ES.restart_initialize(self)
        if is_restart:
            mean, x, p_c, s, vm, pm, b, d, y = self.initialize(is_restart)
            self.d_s *= 2
        return mean, x, p_c, s, vm, pm, b, d, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        mean, x, p_c, s, vm, pm, b, d, y = self.initialize(args)
        while True:
            y_bak = np.copy(y)
            x, y = self.iterate(mean, x, pm, vm, y, b, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, p_c, s, vm, pm, b, d = self._update_distribution(mean, x, p_c, s, vm, pm, b, d, y, y_bak)
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                mean, x, p_c, s, vm, pm, b, d, y = self.restart_initialize(args, mean, x, p_c, s, vm, pm, b, d, y)
        results = self._collect_results(fitness, mean)
        results['p_c'] = p_c
        results['s'] = s
        return results
