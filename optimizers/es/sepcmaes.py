import numpy as np

from optimizers.es.es import ES


class SEPCMAES(ES):
    """Separable Covariance Matrix Adaptation Evolution Strategy (SEPCMAES, sep-CMA-ES).

    Reference
    ---------
    Ros, R. and Hansen, N., 2008, September.
    A simple modification in CMA-ES achieving linear time and space complexity.
    In International Conference on Parallel Problem Solving from Nature (pp. 296-305).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-540-87700-4_30
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.c_c = options.get('c_c', 4 / (self.ndim_problem + 4))
        self.options = options
        self.c_s = None
        self.c_cov = None
        self.d_sigma = None
        self._s_1 = None
        self._s_2 = None

    def _set_c_cov(self):
        c_cov = (1 / self._mu_eff) * (2 / np.power(self.ndim_problem + np.sqrt(2), 2)) + (
            (1 - 1 / self._mu_eff) * np.minimum(1, (2 * self._mu_eff - 1) / (
                np.power(self.ndim_problem + 2, 2) + self._mu_eff)))
        c_cov *= (self.ndim_problem + 2) / 3  # for faster adaptation
        return c_cov

    def _set_d_sigma(self):
        d_sigma = np.maximum((self._mu_eff - 1) / (self.ndim_problem + 1) - 1, 0)
        return 1 + self.c_s + 2 * np.sqrt(d_sigma)

    def initialize(self, is_restart=False):
        self.c_s = self.options.get('c_s', (self._mu_eff + 2) / (self.ndim_problem + self._mu_eff + 3))
        self.c_cov = self.options.get('c_cov', self._set_c_cov())
        self.d_sigma = self.options.get('d_sigma', self._set_d_sigma())
        self._s_1 = 1 - self.c_s
        self._s_2 = np.sqrt(self._mu_eff * self.c_s * (2 - self.c_s))
        z = np.empty((self.n_individuals, self.ndim_problem))  # Gaussian noise for mutation
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        s = np.zeros((self.ndim_problem,))  # evolution path for CSA
        p = np.zeros((self.ndim_problem,))  # evolution path for CMA
        c = np.ones((self.ndim_problem,))  # diagonal elements for covariance matrix
        d = np.ones((self.ndim_problem,))  # diagonal elements for covariance matrix
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return z, x, mean, s, p, c, d, y

    def iterate(self, z=None, x=None, mean=None, d=None, y=None, args=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return z, x, y
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            x[k] = mean + self.sigma * d * z[k]
            y[k] = self._evaluate_fitness(x[k], args)
        return z, x, y

    def _update_distribution(self, z=None, x=None, s=None, p=None, c=None, d=None, y=None):
        order = np.argsort(y)
        zeros = np.zeros((self.ndim_problem,))
        z_w, mean, dz_w = np.copy(zeros), np.copy(zeros), np.copy(zeros)
        for k in range(self.n_parents):
            z_w += self._w[k] * z[order[k]]
            mean += self._w[k] * x[order[k]]  # update distribution mean
            dz = d * z[order[k]]
            dz_w += self._w[k] * dz * dz
        s = self._s_1 * s + self._s_2 * z_w
        if (np.linalg.norm(s) / np.sqrt(1 - np.power(1 - self.c_s, 2 * self._n_generations))) < (
                (1.4 + 2 / (self.ndim_problem + 1)) * self._e_chi):
            h = np.sqrt(self.c_c * (2 - self.c_c)) * np.sqrt(self._mu_eff) * d * z_w
        else:
            h = 0
        p = (1 - self.c_c) * p + h
        c = (1 - self.c_cov) * c + (1 / self._mu_eff) * self.c_cov * p * p + (
                self.c_cov * (1 - 1 / self._mu_eff) * dz_w)
        self.sigma *= np.exp(self.c_s / self.d_sigma * (np.linalg.norm(s) / self._e_chi - 1))
        d = np.sqrt(c)
        return mean, s, p, c, d

    def restart_initialize(self, z=None, x=None, mean=None, s=None, p=None, c=None, d=None, y=None):
        is_restart = ES.restart_initialize(self)
        if is_restart:
            z, x, mean, s, p, c, d, y = self.initialize(is_restart)
            self._n_generations = 0
        return z, x, mean, s, p, c, d, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        z, x, mean, s, p, c, d, y = self.initialize()
        while True:
            self._n_generations += 1
            z, x, y = self.iterate(z, x, mean, d, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, s, p, c, d = self._update_distribution(z, x, s, p, c, d, y)
            self._print_verbose_info(y)
            z, x, mean, s, p, c, d, y = self.restart_initialize(z, x, mean, s, p, c, d, y)
        results = self._collect_results(fitness, mean)
        results['s'] = s
        results['p'] = p
        results['d'] = d
        return results
