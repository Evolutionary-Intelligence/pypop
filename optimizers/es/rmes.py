import numpy as np

from optimizers.es.es import ES
from optimizers.es.r1es import R1ES


class RMES(R1ES):
    def __init__(self, problem, options):
        R1ES.__init__(self, problem, options)
        self.n_evolution_paths = options.get('n_evolution_paths', 2)  # m in Algorithm 2
        self.generation_gap = options.get('generation_gap', self.ndim_problem)  # T in Algorithm 2
        self._a = np.sqrt(1 - self.c_cov)  # for Line 4 in Algorithm 3
        self._a_m = np.power(self._a, self.n_evolution_paths)  # for Line 4 in Algorithm 3
        self._b = np.sqrt(self.c_cov)  # for Line 4 in Algorithm 3

    def initialize(self, args=None):
        x, mean, p, s, y = R1ES.initialize(self, args)
        mp = np.zeros((self.n_evolution_paths, self.ndim_problem))  # multiple evolution paths
        t_hat = np.zeros((self.n_evolution_paths,))
        return x, mean, p, s, mp, t_hat, y

    def iterate(self, x=None, mean=None, p=None, s=None, mp=None, t_hat=None, y=None, args=None):
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

    def _update_distribution(self, x=None, mean=None, p=None, s=None, mp=None, t_hat=None, y=None, y_bak=None):
        mean, p, s = R1ES._update_distribution(self, x, mean, p, s, y, y_bak)
        # update multiple evolution paths
        t_min = np.min(np.diff(t_hat))
        if (t_min > self.generation_gap) or (self._n_generations < self.n_evolution_paths):
            for i in range(self.n_evolution_paths - 1):
                mp[i], t_hat[i] = mp[i + 1], t_hat[i + 1]
        else:
            i_apostrophe = np.argmin(np.diff(t_hat))
            for i in range(i_apostrophe, self.n_evolution_paths - 1):
                mp[i], t_hat[i] = mp[i + 1], t_hat[i + 1]
        mp[-1], t_hat[-1] = p, self._n_generations
        return mean, p, s, mp, t_hat

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        ES.optimize(self, fitness_function)
        fitness = []  # store all fitness generated during evolution
        x, mean, p, s, mp, t_hat, y = self.initialize(args)
        fitness.append(y[0])
        while True:
            y_bak = np.sort(y)
            x, y = self.iterate(x, mean, p, s, mp, t_hat, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y.tolist())
            if self._check_terminations():
                break
            mean, p, s, mp, t_hat = self._update_distribution(x, mean, p, s, mp, t_hat, y, y_bak)
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        results['mean'] = mean
        results['p'] = p
        return results
