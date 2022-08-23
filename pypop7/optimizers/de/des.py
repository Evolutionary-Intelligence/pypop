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
        self.c_d = self.ndim_problem / (self.ndim_problem + 2)
        self.c_c = 1 / np.sqrt(self.ndim_problem)
        self.epsilon = 1e-6
        self.c_epsilon = 2 / (self.ndim_problem ** 2)
        self.h = int(6 + 3 * np.sqrt(self.ndim_problem))  # window size

        self.accu_x = np.empty((0, self.ndim_problem))
        self.accu_delta = np.empty((0, self.ndim_problem))
        self.accu_p = np.empty((0, self.ndim_problem))

    def initialize(self, is_restart=False):
        x = self.rng_initialization.uniform(
            self.initial_lower_boundary, self.initial_upper_boundary,
            (self.n_individuals, self.ndim_problem))
        mean = np.mean(x, 0)
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return x, mean, y

    def iterate(self, x=None, mean=None, y=None, args=None):
        mean_bak = np.copy(mean)
        order = np.argsort(y)[:self.n_parents]
        mean = np.mean(np.copy(x[order]), 0)
        delta = mean - mean_bak
        if self._n_generations == 1:
            p = delta
        else:
            p = (1 - self.c_c) * self.accu_p[-1] + np.sqrt(self.n_parents * self.c_c * (2 - self.c_c)) * delta

        self.accu_x = np.vstack((self.accu_x, x[order]))[-self.h * self.n_parents:]
        self.accu_delta = np.vstack((self.accu_delta, delta))[-self.h:]
        self.accu_p = np.vstack((self.accu_p, p))[-self.h:]

        idx = np.min([self._n_generations, self.h])
        for k in range(self.n_individuals):  # sample population (Line 13)
            if self._check_terminations():
                return x, mean, y

            tau = self.rng_optimization.choice(idx, (3,))  # (Line 14)
            x0 = self.accu_x[(tau[0] * self.n_parents):((tau[0] + 1) * self.n_parents)]
            x1, x2 = self.rng_optimization.choice(x0, (2,))  # (Line 15)

            d = np.sqrt(self.c_d / 2) * (x1 - x2) + \
                np.sqrt(self.c_d) * self.accu_delta[tau[1]] * self.rng_optimization.standard_normal() + \
                np.sqrt(1 - self.c_d) * self.accu_p[tau[2]] * self.rng_optimization.standard_normal() + \
                self.epsilon * np.power((1 - self.c_epsilon), self._n_generations / 2) * \
                self.rng_optimization.standard_normal((self.ndim_problem,))
            x[k] = mean + d
            y[k] = self._evaluate_fitness(x[k], args)
        return x, mean, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, y = self.initialize()
        for k in range(self.n_individuals):
            y[k] = self._evaluate_fitness(x[k], args)
        fitness.extend(y)
        self._n_generations += 1
        while True:
            # sample and evaluate offspring population
            x, mean, y = self.iterate(x, mean, y, args)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness, mean)
        return results
