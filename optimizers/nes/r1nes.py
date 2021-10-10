import time
import numpy as np

from optimizers.nes.nes import NES


class R1NES(NES):
    """Rank-One Natural Evolution Strategy (R1NES).

    Reference
    ---------
    Sun, Y., Schaul, T., Gomez, F. and Schmidhuber, J., 2013, July.
    A linear time natural evolution strategy for non-separable functions.
    In Proceedings of Annual Conference Companion on Genetic and Evolutionary Computation (pp. 61-62).
    https://dl.acm.org/doi/abs/10.1145/2464576.2464608
    https://arxiv.org/abs/1106.1998

    http://schaul.site44.com/code/r1nes.m    (see the official Matlab version)
    """
    def __init__(self, problem, options):
        NES.__init__(self, problem, options)
        if self.eta_mu is None:  # learning rate of mean of Gaussian search distribution (η_μ)
            self.eta_mu = 1
        self.eta_sigma_ = options.get('eta_sigma_', 0.1)
        if options.get('n_individuals') is None:
            self.n_individuals = int(np.floor(8 * np.log(self.ndim_problem)))
        self.upsilon = self._fitness_shaping() + (1 / self.n_individuals)

    def initialize(self):
        s = np.empty((self.n_individuals, self.ndim_problem))  # samples (population / candidate solutions)
        mu = self._initialize_mu()  # mean of Gaussian search distribution
        sigma_ = 0  # [np.exp(sigma_)] == global step-size (sigma)
        # predominant direction: u
        r = 1  # length (== np.exp(c) == ||u||)
        v = self.rng_optimization.standard_normal((self.ndim_problem,))
        v /= np.linalg.norm(v)  # normalized
        u = r * v
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return s, mu, sigma_, r, v, u, y

    def iterate(self, s=None, mu=None, sigma_=None, u=None, y=None, args=None):
        for k in range(self.n_individuals):  # sample population (candidate solutions)
            if self._check_terminations():
                return s, y
            # draw sample (individual)
            s[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            s[k] += u * self.rng_optimization.standard_normal()
            # evaluate fitness
            y[k] = self._evaluate_fitness(mu + np.exp(sigma_) * s[k], args)
        return s, y

    def _compute_gradients(self, s=None, r=None, v=None, y=None):
        upsilon = np.empty((self.n_individuals,))
        upsilon[np.argsort(y)] = np.copy(self.upsilon)
        g_mu = np.dot(upsilon, s)
        s_dot_s, s_dot_v = np.sum(np.power(s, 2), 1), np.dot(s, v)
        g_sigma_ = (s_dot_s - self.ndim_problem) - (np.power(s_dot_v, 2) - 1)
        g_sigma_ = np.dot(g_sigma_, upsilon) / (2 * (self.ndim_problem - 1))
        g_u = (np.power(r, 2) - self.ndim_problem + 2) * np.power(s_dot_v, 2) - (np.power(r, 2) + 1) * s_dot_s
        g_u /= (2 * r * (self.ndim_problem - 1))
        g_u = np.dot(g_u, upsilon) * v + np.dot(s_dot_v / r * upsilon, s)
        g_c = np.dot(g_u, v) / r
        g_v = g_u / r - g_c * v
        return g_mu, g_sigma_, g_u, g_c, g_v

    def _update_distribution(self, mu=None, sigma_=None, r=None, v=None, u=None,
                             g_mu=None, g_sigma_=None, g_u=None, g_c=None, g_v=None):
        mu += self.eta_mu * np.exp(sigma_) * g_mu
        sigma_ += self.eta_sigma_ * g_sigma_
        # learning rate for r, v, u
        eta = np.minimum(0.1, 2 * np.sqrt(np.power(r, 2) / np.dot(g_u, g_u)))
        if g_c < 0:  # multiplicative update
            r *= np.exp(eta * g_c)  # same as [c += eta_c * g_c] since [r == np.exp(c)]
            v = u / r + eta * g_v
            v /= np.linalg.norm(v)
            u = r * v
        else:  # additive update
            u += eta * g_u
            r = np.linalg.norm(u)  # same as [c = np.log(||u||)] since [r == np.exp(c)]
            v = u / r
        return mu, sigma_, r, v, u

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        self.start_time = time.time()
        fitness = []  # store all fitness generated during evolution
        if fitness_function is not None:
            self.fitness_function = fitness_function
        s, mu, sigma_, r, v, u, y = self.initialize()
        while True:
            s, y = self.iterate(s, mu, sigma_, u, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y.tolist())
            if self._check_terminations():
                break
            g_mu, g_sigma_, g_u, g_c, g_v = self._compute_gradients(s, r, v, y)
            mu, sigma_, r, v, u = self._update_distribution(mu, sigma_, r, v, u, g_mu, g_sigma_, g_u, g_c, g_v)
            self.n_generations += 1
            self._print_verbose_info(y)
        if self.record_fitness:
            self._compress_fitness(fitness[:self.n_function_evaluations])
        results = self._collect_results()
        results['mu'] = mu
        results['sigma'] = np.exp(sigma_)
        results['u'] = u
        return results
