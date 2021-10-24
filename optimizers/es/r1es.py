import numpy as np

from optimizers.es.es import ES


class R1ES(ES):
    """Rank One Evolution Strategy (R1ES).

    Reference
    ---------
    Li, Z. and Zhang, Q., 2018.
    A simple yet efficient evolution strategy for large-scale black-box optimization.
    IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
    https://ieeexplore.ieee.org/abstract/document/8080257
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.c_cov = 1 / (3 * np.sqrt(self.ndim_problem) + 5)  # for Line 5 in Algorithm 1
        self.c = 2 / (self.ndim_problem + 7)  # for Line 12 in Algorithm 1
        self.c_s = 0.3  # for Line 15 in Algorithm 1
        self.q_star = 0.3  # for Line 15 in Algorithm 1
        self.d_sigma = 1  # for Line 16 in Algorithm 1
        self._p_1 = 1 - self.c  # for Line 12 in Algorithm 1
        self._p_2 = np.sqrt(self.c * (2 - self.c) * self.mu_eff)  # for Line 12 in Algorithm 1
        self._rr = np.arange(self.n_parents * 2) + 1  # for rule-based success rule

    def initialize(self, args=None):
        x = np.empty((self.n_individuals, self.ndim_problem))  # population
        mu = self._initialize_mu()  # mean of Gaussian search distribution
        p = np.zeros((self.ndim_problem,))  # principal search direction
        s = 0  # cumulative rank rate
        y = np.tile(self._evaluate_fitness(mu, args), (self.n_individuals,))  # fitness
        return x, mu, p, s, y

    def iterate(self, x=None, mu=None, p=None, s=None, y=None, args=None):
        for k in range(self.n_individuals):  # for Line 3 in Algorithm 1
            z = self.rng.standard_normal((self.ndim_problem,))  # for Line 4 in Algorithm 1
            r = self.rng.standard_normal()  # for Line 4 in Algorithm 1
            # for Line 5 in Algorithm 1
            x[k] = mu + self.sigma * (np.sqrt(1 - self.c_cov) * z + np.sqrt(self.c_cov) * r * p)
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _update_distribution(self, x=None, mu=None, p=None, s=None, y=None, y_bak=None):
        order = np.argsort(y)
        y = np.sort(y)
        mu_w = np.zeros((self.ndim_problem,))
        for k in range(self.n_parents):
            mu_w += self.w[k] * x[order[k]]
        p = self._p_1 * p + self._p_2 * (mu_w - mu) / self.sigma  # for Line 12 in Algorithm 1
        mu = mu_w  # for Line 11 in Algorithm 1
        # for rank-based adaptation of mutation strength
        f = np.hstack((y_bak[:self.n_parents], y[:self.n_parents]))
        r = np.argsort(f)
        rr = self._rr[r < self.n_parents] - self._rr[r >= self.n_parents]
        q = np.dot(self.w, rr) / self.n_parents  # for Line 14 in Algorithm 1
        s = (1 - self.c_s) * s + self.c_s * (q - self.q_star)  # for Line 15 in Algorithm 1
        self.sigma *= np.exp(s / self.d_sigma)  # for Line 16 in Algorithm 1
        return mu, p, s

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        ES.optimize(self, fitness_function)
        fitness = []  # store all fitness generated during evolution
        x, mu, p, s, y = self.initialize(args)
        fitness.append(y[0])
        while True:
            y_bak = np.copy(y)
            x, y = self.iterate(x, mu, p, s, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y.tolist())
            if self._check_terminations():
                break
            mu, p, s = self._update_distribution(x, mu, p, s, y, y_bak)
            self.n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        results['mu'] = mu
        results['p'] = p
        return results
