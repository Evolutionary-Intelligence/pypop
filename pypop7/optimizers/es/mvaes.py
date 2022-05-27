import numpy as np

from pypop7.optimizers.es.es import ES


class MVAES(ES):
    """Main Vector Adaptation Evolution Strategies (MVAES, MVA-ES).

    Reference
    ---------
    Poland, J. and Zell, A., 2001, July.
    Main vector adaptation: A CMA variant with linear time and space complexity.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 1050-1055).
    https://dl.acm.org/doi/abs/10.5555/2955239.2955428
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.w_v = options.get('w_v', 3)
        self.c_s = options.get('c_s', 4 / (self.ndim_problem + 4))
        self.c_m = options.get('c_m', 4 / (self.ndim_problem + 4))
        self.c_v = options.get('c_v', 2 / np.power(self.ndim_problem + np.sqrt(2), 2))
        # undefined in the original paper (so we use the default value from CMA-ES)
        self.d_s = 1 / self.c_s + 1
        self._e_chi = np.sqrt(self.ndim_problem - 1 / 2)
        self._c_s_1 = 1 - self.c_s
        self._c_s_2 = np.sqrt(self.c_s * (2 - self.c_s))
        self._c_m_1 = 1 - self.c_m
        self._c_m_2 = np.sqrt(self.c_m * (2 - self.c_m))

    def initialize(self, is_restart=False):
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        p_s = np.zeros((self.ndim_problem,))  # evolution path for global step-size adaptation
        p_m = np.zeros((self.ndim_problem,))  # evolution path for main vector adaptation
        v = np.zeros((self.ndim_problem,))  # main (mutation) vector
        z = np.empty((self.n_individuals, self.ndim_problem))
        z_1 = np.empty((self.n_individuals,))
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return mean, p_s, p_m, v, z, z_1, x, y

    def iterate(self, mean=None, v=None, z=None, z_1=None, x=None, y=None, args=None):
        for k in range(self.n_individuals):
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))  # Line 1
            z_1[k] = self.rng_optimization.standard_normal()  # Line 2
            x[k] = mean + self.sigma * (z[k] + z_1[k] * self.w_v * v)  # Line 3
            y[k] = self._evaluate_fitness(x[k], args)
        return z, z_1, x, y

    def _update_distribution(self, p_s=None, p_m=None, v=None, z=None, z_1=None, x=None, y=None):
        order = np.argsort(y)[:self.n_parents]
        mean = np.mean(x[order], axis=0)
        p_s = self._c_s_1 * p_s + self._c_s_2 * np.mean(z[order], axis=0)  # Line 4
        self.sigma *= np.exp((np.linalg.norm(p_s) - self._e_chi) / (self.d_s * self._e_chi))  # Line 5
        p_m_w = np.zeros((self.ndim_problem,))
        for k in range(self.n_parents):
            p_m_w += (z[order[k]] + z_1[order[k]] * self.w_v * v)
        p_m = self._c_m_1 * p_m + self._c_m_2 * p_m_w / self.n_parents  # Line 6
        v = (1 - self.c_v) * np.sign(np.dot(v, p_m)) * v + self.c_v * p_m  # Line 7
        return mean, p_s, p_m, v

    def restart_initialize(self, mean=None, p_s=None, p_m=None, v=None, z=None, z_1=None, x=None, y=None):
        if ES.restart_initialize(self):
            mean, p_s, p_m, v, z, z_1, x, y = self.initialize(True)
        return mean, p_s, p_m, v, z, z_1, x, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        mean, p_s, p_m, v, z, z_1, x, y = self.initialize()
        while True:
            z, z_1, x, y = self.iterate(mean, v, z, z_1, x, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, p_s, p_m, v = self._update_distribution(p_s, p_m, v, z, z_1, x, y)
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                mean, p_s, p_m, v, z, z_1, x, y = self.restart_initialize(mean, p_s, p_m, v, z, z_1, x, y)
        results = self._collect_results(fitness, mean)
        results['v'] = v
        return results
