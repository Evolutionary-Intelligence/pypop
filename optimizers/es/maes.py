import numpy as np

from optimizers.es.es import ES


class MAES(ES):
    """Matrix Adaptation Evolution Strategy (MAES, (μ/μ_w,λ)-MA-ES).

    Reference
    ---------
    Beyer, H.G., 2020, July.
    Design principles for matrix adaptation evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation Companion (pp. 682-700).
    https://dl.acm.org/doi/abs/10.1145/3377929.3389870

    Loshchilov, I., Glasmachers, T. and Beyer, H.G., 2019.
    Large scale black-box optimization by limited-memory matrix adaptation.
    IEEE Transactions on Evolutionary Computation, 23(2), pp.353-358.
    https://ieeexplore.ieee.org/abstract/document/8410043

    Beyer, H.G. and Sendhoff, B., 2017.
    Simplify your covariance matrix adaptation evolution strategy.
    IEEE Transactions on Evolutionary Computation, 21(5), pp.746-759.
    https://ieeexplore.ieee.org/document/7875115

    https://homepages.fhv.at/hgb/downloads/ForDistributionFastMAES.tar    (see the official Matlab version)
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.c_s = options.get('c_s', self._set_c_s())  # for M10 in Fig. 3
        self.alpha_cov = 2  # for M11 in Fig. 3 (α_cov)
        self.c_1 = options.get('c_1', self._set_c_1())  # for M11 in Fig. 3
        self.c_w = options.get('c_w', self._set_c_w())  # for M11 in Fig. 3 (c_μ)
        self.d_sigma = options.get('d_sigma', self._set_d_sigma())  # for M12 in Fig. 3 (d_σ)
        self._s_1 = 1 - self.c_s  # for M10 in Fig. 3
        self._s_2 = np.sqrt(self._mu_eff * self.c_s * (2 - self.c_s))  # for M10 in Fig. 3
        # for M12 in Fig. 3 (E[||N(0,I)||]: expectation of chi distribution)
        self._e_chi = np.sqrt(self.ndim_problem) * (
            1 - 1 / (4 * self.ndim_problem) + 1 / (21 * np.power(self.ndim_problem, 2)))
        self._fast_version = options.get('_fast_version', False)
        if not self._fast_version:
            self._diag_one = np.diag(np.ones((self.ndim_problem,)))  # for M11 in Fig. 3

    def _set_c_s(self):
        return (self._mu_eff + 2) / (self._mu_eff + self.ndim_problem + 5)

    def _set_c_1(self):
        return self.alpha_cov / (np.power(self.ndim_problem + 1.3, 2) + self._mu_eff)

    def _set_c_w(self):
        return np.minimum(1 - self.c_1, self.alpha_cov * (self._mu_eff + 1 / self._mu_eff - 2) /
                          (np.power(self.ndim_problem + 2, 2) + self.alpha_cov * self._mu_eff / 2))

    def _set_d_sigma(self):
        return 1 + self.c_s + 2 * np.maximum(0, np.sqrt((self._mu_eff - 1) / (self.ndim_problem + 1)) - 1)

    def initialize(self, is_restart=False):  # for M1 in Fig. 3
        self.c_s = self._set_c_s()
        self.c_1 = self._set_c_1()
        self.c_w = self._set_c_w()
        self.d_sigma = self._set_d_sigma()
        self._s_1 = 1 - self.c_s
        self._s_2 = np.sqrt(self._mu_eff * self.c_s * (2 - self.c_s))
        z = np.empty((self.n_individuals, self.ndim_problem))  # Gaussian noise for mutation
        d = np.empty((self.n_individuals, self.ndim_problem))  # search directions
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        s = np.zeros((self.ndim_problem,))  # evolution path
        tm = np.diag(np.ones((self.ndim_problem,)))  # transformation matrix M
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return z, d, mean, s, tm, y

    def iterate(self, z=None, d=None, mean=None, tm=None, y=None, args=None):
        for k in range(self.n_individuals):  # for M3 in Fig. 3 (sample offspring population)
            if self._check_terminations():
                return z, d, y
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))  # for M4 in Fig. 3
            d[k] = np.squeeze(np.dot(tm, z[k][:, np.newaxis]))  # for M5 in Fig. 3
            y[k] = self._evaluate_fitness(mean + self.sigma * d[k], args)  # for M6 in Fig. 3
        return z, d, y

    def _update_distribution(self, z=None, d=None, mean=None, s=None, tm=None, y=None):
        order = np.argsort(y)  # for M8 in Fig. 3
        # for M9, M10, M11 in Fig. 3
        d_w, z_w, zz_w = np.zeros((self.ndim_problem,)), np.zeros((self.ndim_problem,)), None
        if not self._fast_version:
            zz_w = np.zeros((self.ndim_problem, self.ndim_problem))  # for M11 in Fig. 3
        for k in range(self.n_parents):
            d_w += self._w[k] * d[order[k]]
            z_w += self._w[k] * z[order[k]]
            if not self._fast_version:
                zz_w += self._w[k] * np.dot(z[order[k]][:, np.newaxis], z[order[k]][np.newaxis, :])
        # update distribution mean (for M9 in Fig. 3)
        mean += (self.sigma * d_w)
        # update evolution path (s) and transformation matrix (M)
        s = self._s_1 * s + self._s_2 * z_w  # for M10 in Fig. 3
        if not self._fast_version:
            tm_1 = self.c_1 * (np.dot(s[:, np.newaxis], s[np.newaxis, :]) - self._diag_one)
            tm_2 = self.c_w * (zz_w - self._diag_one)
            tm += 0.5 * np.dot(tm, tm_1 + tm_2)  # for M11 in Fig. 3
        else:
            tm *= (1 - 0.5 * (self.c_1 + self.c_w))
            tm += (0.5 * self.c_1) * np.dot(np.dot(tm, s[:, np.newaxis]), s[np.newaxis, :])
            for k in range(self.n_parents):
                tm += (0.5 * self.c_w) * self._w[k] * np.dot(d[order[k]][:, np.newaxis], z[order[k]][np.newaxis, :])
        # update global step-size (for M12 in Fig. 3)
        self.sigma *= np.exp(self.c_s / self.d_sigma * (np.linalg.norm(s) / self._e_chi - 1))
        return mean, s, tm

    def restart_initialize(self, z=None, d=None, mean=None, s=None, tm=None, y=None):
        is_restart = ES.restart_initialize(self)
        if is_restart:
            z, d, mean, s, tm, y = self.initialize(is_restart)
        return z, d, mean, s, tm, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        ES.optimize(self, fitness_function)
        fitness = []  # store all fitness generated during evolution
        z, d, mean, s, tm, y = self.initialize()
        while True:
            z, d, y = self.iterate(z, d, mean, tm, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y.tolist())
            if self._check_terminations():
                break
            mean, s, tm = self._update_distribution(z, d, mean, s, tm, y)
            self._n_generations += 1
            self._print_verbose_info(y)
            z, d, mean, s, tm, y = self.restart_initialize(z, d, mean, s, tm, y)
        results = self._collect_results(fitness, mean)
        results['s'] = s
        return results
