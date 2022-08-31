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
        self.c_d = options.get('c_d', self.n_parents / (self.n_parents + 2))  # for Line 16, 17, 18 in Fig. 4.
        self.c = options.get('c', 1 / np.sqrt(self.ndim_problem))  # for Line 11 in Fig. 4. (c_c)
        self.h = options.get('h', int(6 + 3 * np.sqrt(self.ndim_problem)))  # window size, for Line 14 in Fig. 4. (H)
        self.c_cov = options.get('c_cov', 2 / (self.ndim_problem ** 2))  # for Line 19 in Fig. 4. (c_epsilon)
        self.epsilon = 1e-6  # for Line 19 in Fig. 4.

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            (self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)

        order = np.argsort(y)[:self.n_parents]
        accu_mean = np.array([np.mean(x, 0), np.mean(x[order], 0)])
        accu_p = np.array([accu_mean[-1] - accu_mean[-2]])
        accu_x = x[order]
        self._n_generations += 1
        return x, y, accu_mean, accu_p, accu_x

    def iterate(self, args=None, x=None, y=None, accu_mean=None, accu_p=None, accu_x=None):
        for k in range(self.n_individuals):  # for Line 13 in Fig. 4.
            if self._check_terminations():
                return x, y
            tau = self.rng_optimization.choice(np.min([self._n_generations, self.h]), (3,))  # for Line 14 in Fig. 4.
            x1, x2 = self.rng_optimization.choice(
                accu_x[(tau[0] * self.n_parents):((tau[0] + 1) * self.n_parents)], (2,))
            d = np.sqrt(self.c_d / 2) * (x1 - x2) +\
                np.sqrt(self.c_d) * (accu_mean[tau[1] + 1] - accu_mean[tau[1]]) *\
                self.rng_optimization.standard_normal() +\
                np.sqrt(1 - self.c_d) * accu_p[tau[2]] * self.rng_optimization.standard_normal() +\
                self.epsilon * np.power((1 - self.c_cov), self._n_generations / 2) *\
                self.rng_optimization.standard_normal((self.ndim_problem,))
            x[k] = accu_mean[-1] + d
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _update_distribution(self, x=None, y=None, accu_mean=None, accu_p=None, accu_x=None):
        order = np.argsort(y)[:self.n_parents]
        accu_mean = np.vstack((accu_mean, np.mean(x[order], 0)))[-(self.h + 1):]
        p = (1 - self.c) * accu_p[-1] +\
            np.sqrt(self.n_parents * self.c * (2 - self.c)) * (accu_mean[-1] - accu_mean[-2])  # for Line 11 in Fig. 4.
        accu_p = np.vstack((accu_p, p))[-self.h:]
        accu_x = np.vstack((accu_x, x[order]))[-self.h * self.n_parents:]
        return accu_mean, accu_p, accu_x

    def optimize(self, fitness_function=None, args=None):
        fitness = ES.optimize(self, fitness_function)
        x, y, accu_mean, accu_p, accu_x = self.initialize()
        fitness.extend(y)
        while True:
            x, y = self.iterate(args, x, y, accu_mean, accu_p, accu_x)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            accu_mean, accu_p, accu_x = self._update_distribution(x, y, accu_mean, accu_p, accu_x)
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results
