import time
import numpy as np

from optimizers.es.es import ES


class MAES(ES):
    """Matrix Adaptation Evolution Strategy (MAES, (μ/μ_w, λ)-MA-ES).

    Reference
    ---------
    Beyer, H.G. and Sendhoff, B., 2017.
    Simplify your covariance matrix adaptation evolution strategy.
    IEEE Transactions on Evolutionary Computation, 21(5), pp.746-759.
    https://ieeexplore.ieee.org/document/7875115

    https://homepages.fhv.at/hgb/downloads/ForDistributionFastMAES.tar    (see the official Matlab/Octave version)
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.c_s = (self.mu_eff + 2) / (self.mu_eff + self.ndim_problem + 5)  # for M10 in Fig. 3
        self.s_1 = 1 - self.c_s  # for M10 in Fig. 3
        self.s_2 = np.sqrt(self.mu_eff * self.c_s * (2 - self.c_s))  # for M10 in Fig. 3
        self.alpha_cov = 2  # for M11 in Fig. 3 (α_cov)
        self.c_1 = self.alpha_cov / (np.power(self.ndim_problem + 1.3, 2) + self.mu_eff)  # for M11 in Fig. 3
        self.c_w = np.minimum(1 - self.c_1, self.alpha_cov * (self.mu_eff + 1 / self.mu_eff - 2) / (
                np.power(self.ndim_problem + 2, 2) + self.alpha_cov * self.mu_eff / 2))  # for M11 in Fig. 3
        # for M12 in Fig. 3 (d_σ)
        self.d_sigma = 1 + self.c_s + 2 * np.maximum(
            0, np.sqrt((self.mu_eff - 1) / (self.ndim_problem + 1)) - 1)
        # for M12 in Fig. 3 (E[||N(0,I)||]: expectation of Chi-Square Distribution)
        self.expectation_chi = np.sqrt(self.ndim_problem) * (
            1 - 1 / (4 * self.ndim_problem) - 1 / (21 * np.power(self.ndim_problem, 2)))
        self.diag_one = np.diag(np.ones((self.ndim_problem,)))  # for M11 in Fig. 3

    def initialize(self):  # for M1 in Fig. 3
        z = np.empty((self.n_individuals, self.ndim_problem))  # Gaussian noise for mutation
        d = np.empty((self.n_individuals, self.ndim_problem))  # search directions
        mu = self._initialize_mu()  # mean of Gaussian search distribution
        s = np.zeros((self.ndim_problem,))  # evolution path
        tm = np.diag(np.ones((self.ndim_problem,)))  # transformation matrix M
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return z, d, mu, s, tm, y

    def iterate(self, z=None, d=None, mu=None, s=None, tm=None, y=None, args=None):
        for k in range(self.n_individuals):  # for M3 in Fig. 3 (sample population)
            if self._check_terminations():
                return z, d, y
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))  # for M4 in Fig. 3
            d[k] = np.squeeze(np.dot(tm, z[k][:, np.newaxis]))  # for M5 in Fig. 3
            y[k] = self._evaluate_fitness(mu + self.sigma * d[k], args)  # for M6 in Fig. 3
        return z, d, y

    def _update_distribution(self, z=None, d=None, mu=None, s=None, tm=None, y=None):
        order = np.argsort(y)
        d_w, z_w = np.zeros((self.ndim_problem,)), np.zeros((self.ndim_problem,))  # for M9, M10 in Fig. 3
        zz_w = np.zeros((self.ndim_problem, self.ndim_problem))  # for M11 in Fig. 3
        for k in range(self.n_parents):
            d_w += self.w[k] * d[order[k]]
            z_w += self.w[k] * z[order[k]]
            zz_w += self.w[k] * np.dot(z[order[k]][:, np.newaxis], z[order[k]][np.newaxis, :])
        # update distribution mean (for M9 in Fig. 3)
        mu += (self.sigma * d_w)
        # update evolution path (s) and transformation matrix (M)
        s = self.s_1 * s + self.s_2 * z_w  # for M10 in Fig. 3
        tm_1 = self.c_1 * (np.dot(s[:, np.newaxis], s[np.newaxis, :]) - self.diag_one)
        tm_2 = self.c_w * (zz_w - self.diag_one)
        tm += 0.5 * np.dot(tm, tm_1 + tm_2)  # for M11 in Fig. 3
        # update global step-size (for M12 in Fig. 3)
        self.sigma *= np.exp(self.c_s / self.d_sigma * (np.linalg.norm(s) / self.expectation_chi - 1))
        return mu, s, tm

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        self.start_time = time.time()
        fitness = []  # store all fitness generated during evolution
        if fitness_function is not None:
            self.fitness_function = fitness_function
        z, d, mu, s, tm, y = self.initialize()
        while True:
            z, d, y = self.iterate(z, d, mu, s, tm, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y.tolist())
            if self._check_terminations():
                break
            mu, s, tm = self._update_distribution(z, d, mu, s, tm, y)
            self.n_generations += 1
            self._print_verbose_info(y)
        if self.record_fitness:
            self._compress_fitness(fitness[:self.n_function_evaluations])
        results = self._collect_results()
        results['mu'] = mu
        results['s'] = s
        return results
