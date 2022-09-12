import numpy as np

from pypop7.optimizers.ga.ga import GA
from athena.active import ActiveSubspaces


class ASGA(GA):
    """Active Subspaces extension of the standard GA (ASGA)

    References
    ----------
    Repeat the following paper for `ASGA`:
    Demo, N., Tezzele, M. and Rozza, G., 2021.
    A supervised learning approach involving active subspaces for an efficient genetic algorithm in high-dimensional
    optimization problems.
    SIAM Journal on Scientific Computing, 43(3), pp.B831-B853.
    https://epubs.siam.org/doi/10.1137/20M1345219
    """
    def __init__(self, problem, options):
        GA.__init__(self, problem, options)
        self.crossover_prob = options.get('crossover_prob', 0.5)  # crossover probability
        self.mutate_prob = options.get('mutate_prob', 0.5)  # mutate probability
        self.ndim_subspace = options.get('ndim_subspace', 1)  # the number of active dimensions of subspace
        self.n_b = options.get('b', 2)  # the number of back-mapped points is 2
        self.n_individuals_subspace = int(self.n_individuals / self.n_b)
        self.alpha = options.get('alpha', 1)
        self.n_initial_individuals = options.get('n_initial_individuals', 2000)
        self.n_individuals = options.get('n_individuals', 200)

    def initialize(self):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_initial_individuals, self.ndim_problem))  # initial population
        y = np.empty((self.n_initial_individuals,))  # fitness
        for i in range(self.n_initial_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i])
        self._n_generations += 1
        x_as, y_as = np.copy(x), np.copy(y)
        return x, y, x_as, y_as

    def build_active_space(self, x_as=None, y_as=None):
        active_subspace = ActiveSubspaces(dim=self.ndim_subspace, method='local', n_boot=100)
        active_subspace.fit(inputs=x_as, outputs=y_as)
        return active_subspace

    def select(self, x=None, y=None):
        return x[np.argsort(y)[:self.n_individuals_subspace]]

    def crossover(self, x=None):
        xx = np.copy(x)
        for i in range(self.n_individuals_subspace):
            x1, x2 = self.rng_optimization.choice(x, (2,))
            for j in range(self.ndim_subspace):
                if self.rng_optimization.random() < self.crossover_prob:
                    r = self.rng_optimization.uniform(-1 * self.alpha, 1 + self.alpha)
                    xx[i][j] = (1-r)*x1[j] + r*x2[j]
        return xx

    def mutate(self, x=None):
        for i in range(self.n_individuals_subspace):
            for j in range(self.ndim_subspace):
                if self.rng_optimization.random() < self.mutate_prob:
                    x[i][j] *= (1 + self.rng_optimization.normal(0, 0.1))
        return x

    def iterate(self, x=None, y=None, x_as=None, y_as=None, args=None):
        active_subspace = self.build_active_space(x_as, y_as)  # build active space
        xx = self.select(x, y)  # select
        xx = active_subspace.transform(xx)[0]  # forward (reduction)
        xx = self.crossover(xx)  # crossover (mate)
        xx = self.mutate(xx)  # mutate
        x = active_subspace.inverse_transform(xx, self.n_b)[0]  # backward
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            y[i] = self._evaluate_fitness(x[i], args)
        x_as = np.vstack((x_as, x))
        y_as = np.hstack((y_as, y))
        return x, y, x_as, y_as

    def optimize(self, fitness_function=None):
        fitness = GA.optimize(self, fitness_function)
        x, y, x_as, y_as = self.initialize()
        fitness.extend(y)
        while True:
            x, y, x_as, y_as = self.iterate(x, y, x_as, y_as)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        return self._collect_results(fitness)
