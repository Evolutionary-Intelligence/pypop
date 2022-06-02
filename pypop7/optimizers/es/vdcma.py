import numpy as np

from pypop7.optimizers.es.es import ES


class VDCMA(ES):
    """Linear Covariance Matrix Adaptation (VDCMA, VD-CMA).

    Reference
    ---------
    Akimoto, Y., Auger, A. and Hansen, N., 2014, July.
    Comparison-based natural gradient optimization in high dimension.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 373-380). ACM.
    https://dl.acm.org/doi/abs/10.1145/2576768.2598258

    See the official Python version from Akimoto:
    https://gist.github.com/youheiakimoto/08b95b52dfbf8832afc71dfff3aed6c8
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.options = options
        # for faster adaptation of covariance matrix (c_1, c_mu)
        self.c_factor = options.get('c_factor', np.maximum((self.ndim_problem - 5) / 6, 0.5))
        self.c_c = None
        self.c_1 = None
        self.c_mu = None
        self.c_s = None
        self.d_s = None
        self._v_n = None
        self._v_2 = None
        self._v_ = None
        self._v_p2 = None

    def initialize(self, is_restart=False):
        self.c_c = self.options.get('c_c', (4 + self._mu_eff / self.ndim_problem) / (
                self.ndim_problem + 4 + 2 * self._mu_eff / self.ndim_problem))
        self.c_1 = self.options.get('c_1', self.c_factor * 2 / (
                np.power(self.ndim_problem + 1.3, 2) + self._mu_eff))
        self.c_mu = self.options.get('c_mu', np.minimum(1 - self.c_1, self.c_factor * 2 * (
                self._mu_eff - 2 + 1 / self._mu_eff) / (np.power(self.ndim_problem + 2, 2) + self._mu_eff)))
        self.c_s = self.options.get('c_s', 1 / (2 * np.sqrt(self.ndim_problem / self._mu_eff) + 1))
        self.d_s = self.options.get('d_s', 1 + self.c_s + 2 * np.maximum(0, np.sqrt(
            (self._mu_eff - 1) / (self.ndim_problem + 1)) - 1))
        d = np.ones((self.ndim_problem,))  # diagonal vector for sampling distribution
        # principal search direction (vector) for sampling distribution
        v = self.rng_optimization.standard_normal((self.ndim_problem,)) / np.sqrt(self.ndim_problem)
        p_s = np.zeros((self.ndim_problem,))  # evolution path for step-size adaptation (MCSA)
        p_c = np.zeros((self.ndim_problem,))  # evolution path for covariance matrix adaptation (CMA)
        self._v_n = np.linalg.norm(v)
        self._v_2 = np.power(self._v_n, 2)
        self._v_ = v / self._v_n
        self._v_p2 = np.power(self._v_, 2)
        z = np.empty((self.n_individuals, self.ndim_problem))  # Gaussian noise for mutation
        zz = np.empty((self.n_individuals, self.ndim_problem))  # search directions
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return d, v, p_s, p_c, z, zz, x, mean, y

    def iterate(self, d=None, z=None, zz=None, x=None, mean=None, y=None, args=None):
        for k in range(self.n_individuals):
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            zz[k] = d * (z[k] + (np.sqrt(1 + self._v_2) - 1) * (np.dot(z[k], self._v_) * self._v_))
            x[k] = mean + self.sigma * zz[k]
            y[k] = self._evaluate_fitness(x[k], args)
        return z, zz, x, y

    def _p_q(self, zz, w=0):
        zz_v_ = np.dot(zz, self._v_)
        if isinstance(w, int) and w == 0:
            p = np.power(zz, 2) - self._v_2 / (1 + self._v_2) * (zz_v_ * (zz * self._v_)) - 1
            q = zz_v_ * zz - ((np.power(zz_v_, 2) + 1 + self._v_2) / 2) * self._v_
        else:
            p = np.dot(w, np.power(zz, 2) - self._v_2 / (1 + self._v_2) * (zz_v_ * (zz * self._v_).T).T - 1)
            q = np.dot(w, (zz_v_ * zz.T).T - np.outer((np.power(zz_v_, 2) + 1 + self._v_2) / 2, self._v_))
        return p, q

    def _update_distribution(self, d=None, v=None, p_s=None, p_c=None, zz=None, x=None, y=None):
        order = np.argsort(y)[:self.n_parents]
        # update mean
        mean = np.dot(self._w, x[order])
        # MCSA
        z = np.dot(self._w, zz[order]) / d
        z += (1 / np.sqrt(1 + self._v_2) - 1) * np.dot(z, self._v_) * self._v_
        p_s = (1 - self.c_s) * p_s + np.sqrt(self.c_s * (2 - self.c_s) * self._mu_eff) * z
        p_s_2 = np.dot(p_s, p_s)
        self.sigma *= np.exp((np.sqrt(p_s_2) / self._e_chi - 1) * self.c_s / self.d_s)
        h_s = p_s_2 < (2 + 4 / (self.ndim_problem + 1)) * self.ndim_problem
        # update restricted covariance matrix (d, v)
        p_c = (1 - self.c_c) * p_c + h_s * np.sqrt(
            self.c_c * (2 - self.c_c) * self._mu_eff) * np.dot(self._w, zz[order])

        gamma = 1 / np.sqrt(1 + self._v_2)
        alpha = np.sqrt(np.power(self._v_2, 2) + (1 + self._v_2) / np.max(
            self._v_p2) * (2 - gamma)) / (2 + self._v_2)
        if alpha < 1:
            beta = (4 - (2 - gamma) / np.max(self._v_p2)) / np.power(1 + 2 / self._v_2, 2)
        else:
            alpha, beta = 1, 0
        b = 2 * np.power(alpha, 2) - beta
        a = 2 - (b + 2 * np.power(alpha, 2)) * self._v_p2
        _v_p2_a = self._v_p2 / a

        if self.c_mu == 0:
            p_mu, q_mu = np.zeros((self.ndim_problem,)), np.zeros((self.ndim_problem,))
        else:
            p_mu, q_mu = self._p_q(zz[order] / d, self._w)
        if self.c_1 == 0:
            p_1, q_1 = np.zeros((self.ndim_problem,)), np.zeros((self.ndim_problem,))
        else:
            p_1, q_1 = self._p_q(p_c / d)
        p = self.c_mu * p_mu + h_s * self.c_1 * p_1
        q = self.c_mu * q_mu + h_s * self.c_1 * q_1

        if self.c_mu + self.c_1 > 0:
            r = p - alpha / (1 + self._v_2) * ((2 + self._v_2) * (
                    q * self._v_) - self._v_2 * np.dot(self._v_, q) * self._v_p2)
            s = r / a - b * np.dot(r, _v_p2_a) / (1 + b * np.dot(self._v_p2, _v_p2_a)) * _v_p2_a
            ng_v = q / self._v_n - alpha / self._v_n * ((2 + self._v_2) * (
                    self._v_ * s) - np.dot(s, np.power(self._v_, 2)) * self._v_)
            ng_d = d * s
            up_factor = np.minimum(1, 0.7 * self._v_n / np.sqrt(np.dot(ng_v, ng_v)))
            up_factor = np.minimum(up_factor, 0.7 * (d / np.min(np.abs(ng_d))))
        else:
            ng_v, ng_d, up_factor = np.zeros((self.ndim_problem,)), np.zeros((self.ndim_problem,)), 1
        v += up_factor * ng_v
        d += up_factor * ng_d
        self._v_n = np.linalg.norm(v)
        self._v_2 = np.power(self._v_n, 2)
        self._v_ = v / self._v_n
        self._v_p2 = np.power(self._v_, 2)
        return mean, p_s, p_c, v, d

    def restart_initialize(self, d=None, v=None, p_s=None, p_c=None, z=None, zz=None, x=None, mean=None, y=None):
        if ES.restart_initialize(self):
            d, v, p_s, p_c, z, zz, x, mean, y = self.initialize(True)
        return d, v, p_s, p_c, z, zz, x, mean, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        d, v, p_s, p_c, z, zz, x, mean, y = self.initialize()
        while True:
            z, zz, x, y = self.iterate(d, z, zz, x, mean, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, p_s, p_c, v, d = self._update_distribution(d, v, p_s, p_c, zz, x, y)
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                d, v, p_s, p_c, z, zz, x, mean, y = self.restart_initialize(d, v, p_s, p_c, z, zz, x, mean, y)
        results = self._collect_results(fitness, mean)
        results['d'] = d
        results['v'] = v
        return results
