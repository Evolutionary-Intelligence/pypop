import numpy as np

from optimizers.es.es import ES


class R1ES(ES):
    """Rank-One Evolution Strategy (R1ES).

    Reference
    ---------
    Li, Z. and Zhang, Q., 2018.
    A simple yet efficient evolution strategy for large-scale black-box optimization.
    IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
    https://ieeexplore.ieee.org/abstract/document/8080257
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.c_cov = options.get('c_cov', 1 / (3 * np.sqrt(self.ndim_problem) + 5))  # for Line 5 in Algorithm 1
        self.c = options.get('c', 2 / (self.ndim_problem + 7))  # for Line 12 in Algorithm 1 (c_c)
        self.c_s = options.get('c_s', 0.3)  # for Line 15 in Algorithm 1
        self.q_star = options.get('q_star', 0.3)  # for Line 15 in Algorithm 1
        self.d_sigma = options.get('d_sigma', 1)  # for Line 16 in Algorithm 1
        self._x_1 = np.sqrt(1 - self.c_cov)  # for Line 5 in Algorithm 1
        self._x_2 = np.sqrt(self.c_cov)  # for Line 5 in Algorithm 1
        self._p_1 = 1 - self.c  # for Line 12 in Algorithm 1
        self._p_2 = None  # for Line 12 in Algorithm 1
        self._rr = None  # for rank-based success rule (RSR)

    def initialize(self, args=None, is_restart=False):
        self._p_2 = np.sqrt(self.c * (2 - self.c) * self._mu_eff)
        self._rr = np.arange(self.n_parents * 2) + 1
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        p = np.zeros((self.ndim_problem,))  # principal search direction
        s = 0  # cumulative rank rate
        y = np.tile(self._evaluate_fitness(mean, args), (self.n_individuals,))  # fitness
        return x, mean, p, s, y

    def iterate(self, x=None, mean=None, p=None, y=None, args=None):
        for k in range(self.n_individuals):  # for Line 3 in Algorithm 1
            if self._check_terminations():
                return x, y
            z = self.rng.standard_normal((self.ndim_problem,))  # for Line 4 in Algorithm 1
            r = self.rng.standard_normal()  # for Line 4 in Algorithm 1
            # for Line 5 in Algorithm 1
            x[k] = mean + self.sigma * (self._x_1 * z + self._x_2 * r * p)
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _update_distribution(self, x=None, mean=None, p=None, s=None, y=None, y_bak=None):
        order = np.argsort(y)
        y.sort()  # for Line 10 in Algorithm 1
        # for Line 11 in Algorithm 1
        mean_w = np.zeros((self.ndim_problem,))
        for k in range(self.n_parents):
            mean_w += self._w[k] * x[order[k]]
        p = self._p_1 * p + self._p_2 * (mean_w - mean) / self.sigma  # for Line 12 in Algorithm 1
        mean = mean_w  # for Line 11 in Algorithm 1
        # for rank-based adaptation of mutation strength
        r = np.argsort(np.hstack((y_bak[:self.n_parents], y[:self.n_parents])))
        rr = self._rr[r < self.n_parents] - self._rr[r >= self.n_parents]
        q = np.dot(self._w, rr) / self.n_parents  # for Line 14 in Algorithm 1
        s = (1 - self.c_s) * s + self.c_s * (q - self.q_star)  # for Line 15 in Algorithm 1
        self.sigma *= np.exp(s / self.d_sigma)  # for Line 16 in Algorithm 1
        return mean, p, s

    def restart_initialize(self, args=None, x=None, mean=None, p=None, s=None, y=None, fitness=None):
        is_restart = ES.restart_initialize(self)
        if is_restart:
            x, mean, p, s, y = self.initialize(args, is_restart)
            fitness.append(y[0])
            self.d_sigma *= 2
        return x, mean, p, s, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p, s, y = self.initialize(args)
        fitness.append(y[0])
        while True:
            y_bak = np.copy(y)  # for Line 13 in Algorithm 1
            x, y = self.iterate(x, mean, p, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, p, s = self._update_distribution(x, mean, p, s, y, y_bak)
            self._n_generations += 1
            self._print_verbose_info(y)
            x, mean, p, s, y = self.restart_initialize(args, x, mean, p, s, y, fitness)
        results = self._collect_results(fitness, mean)
        results['p'] = p
        results['s'] = s
        return results
