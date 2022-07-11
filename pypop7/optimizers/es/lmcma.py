import numpy as np

from pypop7.optimizers.es.lmcmaes import LMCMAES


class LMCMA(LMCMAES):
    """Limited Memory Covariance Matrix Adaptation (LMCMA, (μ/μ_w,λ)-LM-CMA).

    Reference
    ---------
    Loshchilov, I., 2017.
    LM-CMA: An alternative to L-BFGS for large-scale black box optimization.
    Evolutionary Computation, 25(1), pp.143-171.
    https://direct.mit.edu/evco/article-abstract/25/1/143/1041/LM-CMA-An-Alternative-to-L-BFGS-for-Large-Scale
    (See Algorithm 7 for details.)

    See the official C++ version from Loshchilov:
    https://sites.google.com/site/ecjlmcma/
    """
    def __init__(self, problem, options):
        LMCMAES.__init__(self, problem, options)
        self.base_m = options.get('base_m', 4)  # base number of direction vectors
        self.period = options.get('period', np.maximum(1, int(np.log(self.ndim_problem))))  # update period
        # self.z_star = options.get('z_star', 0.25)  # target success rate for PSR

    def _a_z(self, z=None, pm=None, vm=None, b=None, start=None, it=None):
        x = np.copy(z)
        for t in range(start, it):
            x = self._a * x + b[self._j[t]] * np.dot(vm[self._j[t]], z) * pm[self._j[t]]
        return x

    def _rademacher(self):
        random = self.rng_optimization.integers(2, size=(self.ndim_problem,))
        random[random == 0] = -1
        return random

    def iterate(self, mean=None, x=None, pm=None, vm=None, y=None, b=None, args=None):
        it = int(np.minimum(self.m, (self._n_generations / self.period)))
        sign, z = 1, np.empty((self.ndim_problem,))  # for mirrored sampling
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            if sign == 1:
                base_m = (10 * self.base_m if k == 0 else self.base_m) * np.abs(
                    self.rng_optimization.standard_normal())
                base_m = it if base_m > it else base_m
                z = self._a_z(self._rademacher(), pm, vm, b, int(it - base_m if it > 1 else 0), it)
            x[k] = mean + sign * self.sigma * z
            y[k] = self._evaluate_fitness(x[k], args)
            sign *= -1  # sample in the opposite direction for mirrored sampling
        return x, y

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
