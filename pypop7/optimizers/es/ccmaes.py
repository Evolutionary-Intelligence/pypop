import numpy as np

from pypop7.optimizers.es.es import ES


class CCMAES(ES):
    """Cholesky-CMA-ES (CCMAES, (μ/μ_w,λ)-Cholesky-CMA-ES).

    Reference
    ---------
    Suttorp, T., Hansen, N. and Igel, C., 2009.
    Efficient covariance matrix update for variable metric evolution strategies.
    Machine Learning, 75(2), pp.167-197.
    https://link.springer.com/article/10.1007/s10994-009-5102-1
    (See Algorithm 4 for details.)
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.c_s = options.get('c_s', np.sqrt(self._mu_eff) / (np.sqrt(self.ndim_problem) + np.sqrt(self._mu_eff)))
        self.d_s = options.get('d_s', 1 + 2 * np.maximum(0, np.sqrt(
            (self._mu_eff - 1) / (self.ndim_problem + 1)) - 1) + self.c_s)
        self.c_c = options.get('c_c', 4 / (self.ndim_problem + 4))
        self.c_cov = options.get('c_cov', 2 / np.power(self.ndim_problem + np.sqrt(2), 2))

    def initialize(self, is_restart=None):
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        z = np.empty((self.n_individuals, self.ndim_problem))  # Gaussian noise for mutation
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        a = np.diag(np.ones(self.ndim_problem,))  # Cholesky factors
        a_i = np.diag(np.ones(self.ndim_problem,))  # inverse of Cholesky factors
        p_s = np.zeros((self.ndim_problem,))  # evolution path for global step-size adaptation
        p_c = np.zeros((self.ndim_problem,))  # evolution path for covariance matrix adaptation
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return mean, z, x, a, a_i, p_s, p_c, y

    def iterate(self, z=None, x=None, mean=None, a=None, y=None, args=None):
        for k in range(self.n_individuals):  # Line 4
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))  # Line 5
            x[k] = mean + self.sigma * np.dot(a, z[k])  # Line 6
            y[k] = self._evaluate_fitness(x[k], args)
        return z, x, y

    def _update_distribution(self, z=None, x=None, a=None, a_i=None, p_s=None, p_c=None, y=None):
        order = np.argsort(y)[:self.n_parents]
        mean, z_w = np.dot(self._w, x[order]), np.dot(self._w, z[order])  # Line 3 and 7
        p_c = (1 - self.c_c) * p_c + np.sqrt(self.c_c * (2 - self.c_c) * self._mu_eff) * np.dot(a, z_w)  # Line 8
        v = np.dot(a_i, p_c)  # Line 9
        v_norm = np.dot(v, v)  # (||v||)^2
        s_v_norm = np.sqrt(1 + self.c_cov / (1 - self.c_cov) * v_norm)
        a_i = (a_i - (1 - 1 / s_v_norm) * np.dot(v[:, np.newaxis], np.dot(v[np.newaxis, :], a_i)) / v_norm
               ) / np.sqrt(1 - self.c_cov)  # Line 10
        a = np.sqrt(1 - self.c_cov) * (a + (s_v_norm - 1) * np.dot(
            p_c[:, np.newaxis], v[np.newaxis, :]) / v_norm)  # Line 11
        p_s = (1 - self.c_s) * p_s + np.sqrt(self.c_s * (2 - self.c_s) * self._mu_eff) * z_w  # Line 12
        self.sigma *= np.exp(self.c_s / self.d_s * (np.linalg.norm(p_s) / self._e_chi - 1))  # Line 13
        return mean, a, a_i, p_s, p_c

    def restart_initialize(self, mean=None, z=None, x=None, a=None, a_i=None, p_s=None, p_c=None, y=None):
        is_restart = ES.restart_initialize(self)
        if is_restart:
            mean, z, x, a, a_i, p_s, p_c, y = self.initialize(True)
        return mean, z, x, a, a_i, p_s, p_c, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        mean, z, x, a, a_i, p_s, p_c, y = self.initialize()
        while True:
            # sample and evaluate offspring population
            z, x, y = self.iterate(z, x, mean, a, y, args)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, a, a_i, p_s, p_c = self._update_distribution(z, x, a, a_i, p_s, p_c, y)
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                mean, z, x, a, a_i, p_s, p_c, y = self.restart_initialize(mean, z, x, a, a_i, p_s, p_c, y)
        results = self._collect_results(fitness, mean)
        return results
