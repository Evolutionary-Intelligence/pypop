import numpy as np

from pypop7.optimizers.ep.ep import EP


class CEP(EP):
    """Classical Evolutionary Programming with self-adaptive mutation (CEP).

    References
    ----------
    Yao, X., Liu, Y. and Lin, G., 1999.
    Evolutionary programming made faster.
    IEEE Transactions on Evolutionary Computation, 3(2), pp.82-102.
    https://ieeexplore.ieee.org/abstract/document/771163
    """
    def __init__(self, problem, options):
        EP.__init__(self, problem, options)
        self.sigma = options.get('sigma')  # initial global step-size
        self.q = options.get('q', 10)  # number of opponents for pairwise comparisons
        # two learning rate factors of individual step-size
        self.tau = options.get('tau', 1.0 / np.sqrt(2.0*np.sqrt(self.ndim_problem)))
        self.tau_apostrophe = options.get('tau_apostrophe', 1.0 / np.sqrt(2.0*self.ndim_problem))

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))
        sigmas = self.sigma*np.ones((self.n_individuals, self.ndim_problem))  # eta (η)
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        offspring_x = np.empty((self.n_individuals, self.ndim_problem))
        offspring_sigmas = np.empty((self.n_individuals, self.ndim_problem))  # eta (η)
        offspring_y = np.empty((self.n_individuals,))
        return x, sigmas, y, offspring_x, offspring_sigmas, offspring_y

    def iterate(self, x=None, sigmas=None, y=None,
                offspring_x=None, offspring_sigmas=None, offspring_y=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, sigmas, y, offspring_x, offspring_sigmas, offspring_y
            for j in range(self.ndim_problem):
                n_j = self.rng_optimization.standard_normal()
                offspring_x[i][j] = x[i][j] + sigmas[i][j]*n_j
                offspring_sigmas[i][j] = sigmas[i][j]*np.exp(
                    self.tau_apostrophe*self.rng_optimization.standard_normal() + self.tau*n_j)
            offspring_y[i] = self._evaluate_fitness(offspring_x[i])
        new_x = np.vstack((offspring_x, x))
        new_sigmas = np.vstack((offspring_sigmas, sigmas))
        new_y = np.hstack((offspring_y, y))
        n_win = np.zeros((2*self.n_individuals,))  # number of win
        for i in range(2*self.n_individuals):
            for j in self.rng_optimization.choice(2*self.n_individuals, self.q):
                if new_y[i] <= new_y[j]:
                    n_win[i] += 1
        order = np.argsort(n_win)[::-1]  # in decreasing order for minimization
        for i in range(self.n_individuals):
            x[i] = new_x[order[i]]
            sigmas[i] = new_sigmas[order[i]]
            y[i] = new_y[order[i]]
        return x, sigmas, y, offspring_x, offspring_sigmas, offspring_y

    def optimize(self, fitness_function=None, args=None):
        fitness = EP.optimize(self, fitness_function)
        x, sigmas, y, offspring_x, offspring_sigmas, offspring_y = self.initialize()
        fitness.extend(y)
        while True:
            x, sigmas, y, offspring_x, offspring_sigmas, offspring_y = self.iterate(
                x, sigmas, y, offspring_x, offspring_sigmas, offspring_y)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        return self._collect_results(fitness)
