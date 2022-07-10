import numpy as np

from pypop7.optimizers.es.es import ES


class LMCMA(ES):
    """Limited Memory Covariance Matrix Adaptation (LMCMA, (μ/μ_w,λ)-LM-CMA).

    Reference
    ---------
    (See Algorithm 7 for details.)

    See the official C++ version from Loshchilov:

    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.c_s = options.get('c_s', 0.3)  # learning rate for population success rule (PSR)
        self.d_s = options.get('d_s', 1)
        self.m = options.get('m', self.n_individuals)  # number of direction vectors
        self.base_m = options.get('base_m', 4)  # base number of direction vectors
        self.n_steps = options.get('n_steps', self.m)  # target number of generations between vectors
        self.c_c = options.get('c_c', 1 / self.m)  # learning rate for evolution path
        self.c_1 = 1 / (10 * np.log(self.ndim_problem + 1))  # learning rate for covariance matrix adaptation
        self.z_star = 0.3  # target success rate for PSR
        self.period = np.maximum(1, int(np.log(self.ndim_problem)))
        self._a = np.sqrt(1 - self.c_1)
        self._c = 1 / np.sqrt(1 - self.c_1)
        self._bd_1 = np.sqrt(1 - self.c_1)
        self._bd_2 = self.c_1 / (1 - self.c_1)
        self._p_c_1 = 1 - self.c_c
        self._p_c_2 = None
        self._rr = None
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

    def _a_z(self, z=None, pm=None, vm=None, b=None, start=None):
        x = np.copy(z)
        for t in range(start, int(np.minimum(self.m, self._n_generations / self.period))):
            x = self._a * x + b[self._j[t]] * np.dot(vm[self._j[t]], z) * pm[self._j[t]]
        return x

    def _rademacher(self):
        random = self.rng_optimization.integers(2, size=(self.ndim_problem,))
        random[random == 0] = -1
        return random

    def iterate(self, mean=None, x=None, pm=None, vm=None, y=None, b=None, args=None):
        it = np.minimum(self.m, (self._n_generations / self.period) + 1)
        sign, z = 1, np.empty((self.ndim_problem,))  # for mirrored sampling
        for k in range(self.n_individuals):
            if sign == 1:
                base_m = 10 * self.base_m if k == 0 else self.base_m
                base_m *= np.abs(self.rng_optimization.standard_normal())
                base_m = it if base_m > it else base_m
                z = self._a_z(self._rademacher(), pm, vm, b, int(it - base_m if it > 1 else 0))
            x[k] = mean + sign * self.sigma * z
            y[k] = self._evaluate_fitness(x[k], args)
            sign *= -1  # sample in the opposite direction for mirrored sampling
        return x, y

    def _a_inv_z(self, v=None, vm=None, d=None, i=None):
        x = np.copy(v)
        for t in range(0, i):
            x = self._c * x - d[self._j[t]] * np.dot(vm[self._j[t]], x) * vm[self._j[t]]
        return x

    def _update_distribution(self, mean=None, x=None, p_c=None, s=None, vm=None, pm=None,
                             b=None, d=None, y=None, y_bak=None):
        mean_bak = np.dot(self._w, x[np.argsort(y)[:self.n_parents]])
        p_c = self._p_c_1 * p_c + self._p_c_2 * (mean_bak - mean) / self.sigma
        if self._n_generations % self.period == 0:
            _n_generations = int(self._n_generations / self.period)
            if _n_generations < self.m:
                i_min, self._j[_n_generations] = _n_generations, _n_generations
            elif self.m > 1:
                i_min, d_min = 1, self._l[self._j[1]] - self._l[self._j[0]]
                for j in range(2, self.m):
                    d_cur = self._l[self._j[j]] - self._l[self._j[j - 1]]
                    if d_cur < d_min:
                        d_min, i_min = d_cur, j
                # if all pairwise distances exceed self.n_steps, start from 0
                i_min = 0 if d_min >= self.n_steps else i_min
                updated = self._j[i_min]
                for j in range(i_min, self.m - 1):
                    self._j[j] = self._j[j + 1]
                self._j[self.m - 1] = updated
            else:
                i_min = 0
            pm[self._j[np.minimum(self.m - 1, _n_generations)]] = p_c
            self._l[self._j[np.minimum(self.m - 1, _n_generations)]] = self._n_generations
            for i in range(i_min, np.minimum(self.m, _n_generations + 1)):
                vm[self._j[i]] = self._a_inv_z(pm[self._j[i]], vm, d, i)
                v_n = np.dot(vm[self._j[i]], vm[self._j[i]])
                bd_3 = np.sqrt(1 + self._bd_2 * v_n)
                b[self._j[i]] = self._bd_1 / v_n * (bd_3 - 1)
                d[self._j[i]] = 1 / (self._bd_1 * v_n) * (1 - 1 / bd_3)
        y.sort()
        if self._n_generations > 0:
            r = np.argsort(np.hstack((y, y_bak)))
            z_psr = np.sum(self._rr[r < self.n_individuals] - self._rr[r >= self.n_individuals])
            z_psr = z_psr / np.power(self.n_individuals, 2) - self.z_star
            s = (1 - self.c_s) * s + self.c_s * z_psr
            self.sigma *= np.exp(s / self.d_s)
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
