import numpy as np

from optimizers.es.es import ES
from optimizers.es.r1es import R1ES


class RMES(R1ES):
    """Rank-M Evolution Strategy (RMES).

    Reference
    ---------
    Li, Z. and Zhang, Q., 2018.
    A simple yet efficient evolution strategy for large-scale black-box optimization.
    IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
    https://ieeexplore.ieee.org/abstract/document/8080257
    """
    def __init__(self, problem, options):
        R1ES.__init__(self, problem, options)
        self.n_evolution_paths = options.get('n_evolution_paths', 2)  # m in Algorithm 2
        self.generation_gap = options.get('generation_gap', self.ndim_problem)  # T in Algorithm 2
        self._a = np.sqrt(1 - self.c_cov)  # for Line 4 in Algorithm 3
        self._a_m = np.power(self._a, self.n_evolution_paths)  # for Line 4 in Algorithm 3
        self._b = np.sqrt(self.c_cov)  # for Line 4 in Algorithm 3

    def initialize(self, args=None, is_restart=False):
        x, mean, p, s, y = R1ES.initialize(self, args, is_restart)
        mp = np.zeros((self.n_evolution_paths, self.ndim_problem))  # multiple evolution paths
        t_hat = np.zeros((self.n_evolution_paths,))  # for Algorithm 2
        return x, mean, p, s, mp, t_hat, y

    def iterate(self, x=None, mean=None, mp=None, y=None, args=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            z = self.rng.standard_normal((self.ndim_problem,))
            sum_p = np.zeros((self.ndim_problem,))
            for i in np.arange(self.n_evolution_paths) + 1:
                r = self.rng.standard_normal()
                sum_p += np.power(self._a, self.n_evolution_paths - i) * r * mp[i - 1]
            x[k] = mean + self.sigma * (self._a_m * z + self._b * sum_p)
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _update_distribution(self, x=None, mean=None, p=None, s=None,
                             mp=None, t_hat=None, y=None, y_bak=None):
        mean, p, s = R1ES._update_distribution(self, x, mean, p, s, y, y_bak)
        # update multiple evolution paths
        t_min = np.min(np.diff(t_hat))  # Line 2 in Algorithm 2 (T_min)
        if (t_min > self.generation_gap) or (self._n_generations < self.n_evolution_paths):
            for i in range(self.n_evolution_paths - 1):
                mp[i], t_hat[i] = mp[i + 1], t_hat[i + 1]
        else:
            i_apostrophe = np.argmin(np.diff(t_hat))  # Line 6 in Algorithm 2 (i')
            for i in range(i_apostrophe, self.n_evolution_paths - 1):
                mp[i], t_hat[i] = mp[i + 1], t_hat[i + 1]
        mp[-1], t_hat[-1] = p, self._n_generations
        return mean, p, s, mp, t_hat

    def restart_initialize(self, args=None, x=None, mean=None, p=None, s=None,
                           mp=None, t_hat=None, y=None, fitness=None):
        is_restart = ES.restart_initialize(self)
        if is_restart:
            x, mean, p, s, mp, t_hat, y = self.initialize(args, is_restart)
            fitness.append(y[0])
            self.d_sigma *= 2
        return x, mean, p, s, mp, t_hat, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p, s, mp, t_hat, y = self.initialize(args)
        fitness.append(y[0])
        while True:
            y_bak = np.copy(y)
            x, y = self.iterate(x, mean, mp, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, p, s, mp, t_hat = self._update_distribution(x, mean, p, s, mp, t_hat, y, y_bak)
            self._n_generations += 1
            self._print_verbose_info(y)
            x, mean, p, s, mp, t_hat, y = self.restart_initialize(
                args, x, mean, p, s, mp, t_hat, y, fitness)
        results = self._collect_results(fitness, mean)
        results['p'] = p
        results['s'] = s
        return results
