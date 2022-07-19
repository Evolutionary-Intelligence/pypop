import random

import numpy as np

from pypop7.optimizers.eda.eda import EDA


def corr(x):
    c = np.cov(x, rowvar=False)
    dim = len(c)
    stds = np.std(x, axis=0)
    for i in range(dim):
        for j in range(dim):
            c[i][j] /= (stds[i] * stds[j])
    return c


def mvnrnd(mean, cov, n):
    cov = cov + np.diag(np.repeat(1e-30, len(cov)))
    L = np.linalg.cholesky(cov)
    x = np.dot(np.random.randn(n, len(mean)), L) + np.tile(mean, (n, 1))
    return x


class EDAMCC(EDA):
    """Estimation of distribution algorithms framework with model complexity control(EDA-MCC)
        Reference
        --------------
        W. Dong, T. Chen, P. Tino, X. Yao
        Scaling Up Estimation of Distribution Algorithms for Continuous Optimization
        IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 17, NO. 6, DECEMBER 2013
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6461934
    """
    def __init__(self, problem, options):
        EDA.__init__(self, problem, options)
        self.x = None
        self.n_parents = int(0.2 * self.n_individuals)
        self.m_corr = options.get('m_corr')
        if self.m_corr is None:
            self.m_corr = int(0.5 * self.n_parents)
        self.theta = options.get('theta', 0.3)
        self.c = options.get('c', 3)

    def initialize(self):
        x = np.empty((self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            x[i] = self._initialize_x()
            y[i] = self._evaluate_fitness(x[i])
        return x, y

    def iterate(self, x, y):
        x_fit = np.empty((self.n_parents, self.ndim_problem))
        new_x = np.empty((self.n_individuals, self.ndim_problem))
        order = np.argsort(y)
        for i in range(self.n_parents):
            x_fit[i] = x[order[i]]
        rand_arr = np.arange(x_fit.shape[0])
        np.random.shuffle(rand_arr)
        x_corr = x_fit[rand_arr[0: self.m_corr]]
        cor = corr(x_corr)
        w, s = [], []
        for i in range(self.ndim_problem):
            judge = True
            for j in range(self.m_corr):
                if i != j and np.abs(cor[i][j]) > self.theta:
                    judge = False
                    break
            if judge is True:
                w.append(i)
            else:
                s.append(i)

        # Weakly dependent variable identification
        mean_weak = np.mean(x_fit[:, w], axis=0)
        std_weak = np.std(x_fit[:, w], axis=0)
        new_x[:, w] = np.tile(mean_weak, (self.n_individuals, 1)) \
                      + np.random.randn(self.n_individuals, len(w)) * np.tile(std_weak, (self.n_individuals, 1))

        # subspace modeling
        while len(s) != 0:
            idx = random.sample(s, min(self.c, len(s)))
            for i in range(len(idx)):
                for j in range(len(s)):
                    if idx[i] == s[j]:
                        s.remove(s[j])
                        break
            cur_mean = np.mean(x_fit[:, idx], axis=0)
            if len(idx) > 1:
                cur_cov = np.cov(x_fit[:, idx], rowvar=False)
                new_x[:, idx] = mvnrnd(cur_mean, cur_cov, self.n_individuals)
            else:
                cur_std = np.std(x_fit[:, idx], axis=0)
                new_x[:, idx] = self.rng_optimization.normal(cur_mean, cur_std, size=(self.n_individuals, 1))

        for i in range(self.n_individuals):
            new_x[i] = np.clip(new_x[i], self.lower_boundary, self.upper_boundary)
            if self._check_terminations():
                return new_x, y
            y[i] = self._evaluate_fitness(new_x[i])
        order1 = np.argsort(y)
        new_x[order1[-1]] = x[order[0]]
        y[order1[-1]] = self._evaluate_fitness(new_x[order1[-1]])
        return new_x, y

    def optimize(self, fitness_function=None):
        fitness = EDA.optimize(self, fitness_function)
        x, y = self.initialize()
        while True:
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            x, y = self.iterate(x, y)
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results
