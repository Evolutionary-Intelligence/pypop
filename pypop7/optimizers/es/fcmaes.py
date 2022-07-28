import numpy as np

from pypop7.optimizers.es.es import ES


class FCMAES(ES):
    """Fast Covariance Matrix Adaptation Evolution Strategies(FCMAES, (μ/μ_w,λ)-Fast CMA-ES)
        Reference
        ---------------
        Z. Li, Q. Zhang, X. Lin, H. zhen
        Fast Covariance Matrix Adaptation for Large-Scale Black-Box Optimization
        IEEE Transaction on Cybernetics vol.50 No.5 May 2020
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8533604
    """
    def __init__(self, prolblem, options):
        ES.__init__(self, prolblem, options)
        self.n_evolution_paths = self.n_individuals
        self.c = 2 / (self.ndim_problem + 5)
        self.c_1 = 1 / (3 * np.sqrt(self.ndim_problem) + 5)
        self.c_sigma = 0.3
        self.d_sigma = 1
        self.q_star = 0.27
        self.T = self.ndim_problem
        self.sigma_threshold = 1e-12
        self.fitness_diff = 1e-12
        self.fitness_rel_diff = 1e-6
        self.s_1 = 1 - self.c_1
        self.s_2 = np.sqrt((1 - self.c_1) * self.c_1)
        self.s_3 = np.sqrt(self.c_1)
        self.d_1 = 1 - self.c
        self.d_2 = np.sqrt(self.c * (2 - self.c) * self._mu_eff)

    def initialize(self, is_restart=False):
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        x = np.empty((self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals, ))
        p = np.zeros((self.ndim_problem, ))
        p_hat = np.zeros((self.n_evolution_paths, self.ndim_problem))
        s = 0
        f = np.inf * np.ones((self.n_parents, ))
        return mean, x, y, p, p_hat, s, f

    def iterate(self, mean, x, y, p, p_hat):
        for i in range(self.n_individuals):
            z = self.rng_optimization.standard_normal((self.ndim_problem, ))
            r1 = self.rng_optimization.standard_normal()
            r2 = self.rng_optimization.standard_normal()
            x[i] = mean + self.sigma * (self.s_1 * z + self.s_2 * r1 * p_hat[i] + self.s_3 * r2 * p)
            y[i] = self._evaluate_fitness(x[i])
        return x, y

    def _update_distribution(self, mean, x, y, p, p_hat, s, f):
        new_mean = 0
        new_f = np.empty((self.n_parents, ))
        order = np.argsort(y)
        r1, r2 = np.empty((self.n_parents, )), np.empty((self.n_parents, ))
        for i in range(self.n_parents):
            new_mean += self._w[i] * x[order[i]]
            new_f[i] = y[order[i]]
        p = self.d_1 * p + self.d_2 * (new_mean - mean) / self.sigma
        mean = new_mean
        if self._n_generations % self.T == 0:
            for i in range(self.n_evolution_paths - 1):
                p_hat[i] = p_hat[i+1]
            p_hat[-1] = p
        F = np.append(f, new_f, axis=0)
        F = np.sort(F)
        p1, p2 = 0, 0
        for i in range(len(F)):
            if p1 != self.n_parents and f[p1] == F[i]:
                r1[p1] = i
                p1 += 1
            elif new_f[p2] == F[i]:
                r2[p2] = i
                p2 += 1
        assert p1 == self.n_parents
        assert p2 == self.n_parents
        q = 0
        for i in range(self.n_parents):
            q += self._w[i] * (r1[i] - r2[i])
        q /= self.n_parents
        f = new_f
        s = (1 - self.c_sigma) * s + self.c_sigma * (q - self.q_star)
        self.sigma *= np.exp(s / self.d_sigma)
        return mean, p, p_hat, s, f

    def restart_initialize(self, args=None, mean=None, x=None, y=None, p=None, p_hat=None,
                           s=None, f=None):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2, is_restart_3 = self.sigma < self.sigma_threshold, False, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
            is_restart_3 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) /\
                           self._fitness_list[-1] < self.fitness_rel_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2) or bool(is_restart_3)
        if is_restart:
            self.n_restart += 1
            self.sigma = np.copy(self._sigma_bak)
            self.n_individuals *= 2
            self.n_parents = int(self.n_individuals / 2)
            self.n_evolution_paths = self.n_individuals
            w_base, w = np.log((self.n_individuals + 1) / 2), np.log(np.arange(self.n_parents) + 1)
            self._w = (w_base - w) / (self.n_parents * w_base - np.sum(w))
            self._mu_eff = 1 / np.sum(np.power(self._w, 2))
            self._fitness_list = [np.Inf]
            self.d_sigma *= 2
            mean, x, y, p, p_hat, s, f = self.initialize(args, is_restart)
        return mean, x, y, p, p_hat, s, f

    def optimize(self, fitness_function=None, args=None):
        fitness = ES.optimize(self, fitness_function)
        mean, x, y, p, p_hat, s, f = self.initialize()
        while True:
            x, y = self.iterate(mean, x, y, p, p_hat)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, p, p_hat, s, f = self._update_distribution(mean, x, y, p, p_hat, s, f)
            self._n_generations += 1
            self._print_verbose_info(y)
            mean, x, y, p, p_hat, s, f = self.restart_initialize(args, mean, x, y, p, p_hat, s, f)
        results = self._collect_results(fitness, mean)
        return results
