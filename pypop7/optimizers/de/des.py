import numpy as np

from pypop7.optimizers.es.es import ES


class DES(ES):
    """Differential Evolution Strategy (DES)

    Reference
    ---------
    Arabas, J., and Jagodziński, D. 2020.
    Toward a matrix-free covariance matrix adaptation evolution strategy.
    IEEE Transactions on Evolutionary Computation, 24(1), 84–98.
    https://doi.org/10.1109/TEVC.2019.2907266
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.c_d = options.get('c', self.ndim_problem / (self.ndim_problem + 2))  # for Line 16, 17, 18 in Fig. 4.
        self.c = options.get('c', 1 / np.sqrt(self.ndim_problem))  # for Line 11 in Fig. 4. (c_c)
        self.h = options.get('h', int(6 + 3 * np.sqrt(self.ndim_problem)))  # window size, for Line 14 in Fig. 4. (H)
        self.c_cov = options.get('c_cov', 2 / (self.ndim_problem ** 2))  # for Line 19 in Fig. 4. (c_epsilon)
        self.epsilon = 1e-6  # for Line 19 in Fig. 4.

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        mean = np.mean(x, 0)
        accu_p = np.empty((0, self.ndim_problem))
        accu_d = np.empty((0, self.ndim_problem))
        accu_x = np.empty((0, self.ndim_problem))
        self._n_generations += 1
        return x, mean, accu_p, accu_d, accu_x, y

    def iterate(self, x=None, mean=None, accu_p=None, accu_d=None, accu_x=None, y=None, args=None):
        for k in range(self.n_individuals):  # for Line 13 in Fig. 4.
            if self._check_terminations():
                return x, y
            tau = self.rng_optimization.choice(np.min([self._n_generations, self.h]), (3,))  # for Line 14 in Fig. 4.
            x1, x2 = self.rng_optimization.choice(
                accu_x[(tau[0] * self.n_parents):((tau[0] + 1) * self.n_parents)], (2,))
            d = np.sqrt(self.c_d / 2) * (x1 - x2) + \
                np.sqrt(self.c_d) * accu_d[tau[1]] * self.rng_optimization.standard_normal() + \
                np.sqrt(1 - self.c_d) * accu_p[tau[2]] * self.rng_optimization.standard_normal() + \
                self.epsilon * np.power((1 - self.c_cov), self._n_generations / 2) * \
                self.rng_optimization.standard_normal((self.ndim_problem,))
            x[k] = mean + d
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _update_distribution(self, x=None, mean=None, accu_p=None, accu_d=None, accu_x=None, y=None):
        mean_bak = np.copy(mean)
        order = np.argsort(y)[:self.n_parents]
        mean = np.mean(x[order], 0)
        d = mean - mean_bak  # for Line 7 in Fig. 4.
        if self._n_generations == 1:
            p = d  # for Line 9 in Fig. 4.
        else:
            p = (1 - self.c) * accu_p[-1] + np.sqrt(self.n_parents * self.c * (2 - self.c)) * d
        accu_p = np.vstack((accu_p, p))[-self.h:]
        accu_d = np.vstack((accu_d, d))[-self.h:]
        accu_x = np.vstack((accu_x, x[order]))[-self.h * self.n_parents:]
        return mean, accu_p, accu_d, accu_x

    def optimize(self, fitness_function=None, args=None):
        fitness = ES.optimize(self, fitness_function)
        x, mean, accu_p, accu_d, accu_x, y = self.initialize()
        fitness.extend(y)
        while True:
            mean, accu_p, accu_d, accu_x = self._update_distribution(x, mean, accu_p, accu_d, accu_x, y)
            x, y = self.iterate(x, mean, accu_p, accu_d, accu_x, y, args)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results
