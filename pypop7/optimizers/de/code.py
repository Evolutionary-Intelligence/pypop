import numpy as np

from pypop7.optimizers.de.de import DE


class CODE(DE):
    """Composite Differential Evolution (CoDE).

    Reference
    ---------
    Wang, Y., Cai, Z., and Zhang, Q. 2011.
    Differential evolution with composite trial vector generation strategies and control parameters.
    IEEE Transactions on Evolutionary Computation, 15(1), pp.55–66.
    https://doi.org/10.1109/TEVC.2010.2087271
    """

    def __init__(self, problem, options):
        DE.__init__(self, problem, options)
        self.pool = [[1.0, 0.1], [1.0, 0.9], [0.8, 0.2]]  # [f_mu, p_cr], control parameter settings pool
        self.boundary = options.get('boundary', False)

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            (self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        self._n_generations += 1
        return x, y

    def bound(self, x=None):
        if not self.boundary:
            return x
        for k in range(self.n_individuals):
            idx = np.array(x[k] < self.lower_boundary)
            if idx.any():
                x[k][idx] = np.minimum(self.upper_boundary, 2 * self.lower_boundary - x[k])[idx]
            idx = np.array(x[k] > self.upper_boundary)
            if idx.any():
                x[k][idx] = np.maximum(self.lower_boundary, 2 * self.upper_boundary - x[k])[idx]
        return x

    def mutate(self, x=None):
        x1_mu = np.empty((self.n_individuals,  self.ndim_problem))
        x2_mu = np.empty((self.n_individuals, self.ndim_problem))
        x3_mu = np.empty((self.n_individuals, self.ndim_problem))
        # randomly selected from the parameter candidate pool
        f_p = self.rng_optimization.choice(self.pool, (self.n_individuals, 3))
        for k in range(self.n_individuals):
            r = self.rng_optimization.choice(np.setdiff1d(np.arange(self.n_individuals), k), (3,), False)
            x1_mu[k] = x[r[0]] + f_p[k, 0, 0] * (x[r[1]] - x[r[2]])  # rand/1/bin

            # In order to improve the search ability, the first scaling factor is randomly chosen from 0 to 1
            r = self.rng_optimization.choice(np.setdiff1d(np.arange(self.n_individuals), k), (5,), False)
            x2_mu[k] = x[r[0]] + self.rng_optimization.random() * (x[r[1]] - x[r[2]]) +\
                       f_p[k, 1, 0] * (x[r[3]] - x[r[4]])  # rand/2/bin

            r = self.rng_optimization.choice(np.setdiff1d(np.arange(self.n_individuals), k), (3,), False)
            x3_mu[k] = x[k] + self.rng_optimization.random() * (x[r[0]] - x[k]) +\
                       f_p[k, 2, 0] * (x[r[1]] - x[r[2]])  # current-to-rand/1
        return x1_mu, x2_mu, x3_mu, f_p

    def crossover(self, x_mu=None, x=None, p_cr=None):
        x_cr = np.copy(x)
        for k in range(self.n_individuals):
            for i in range(self.ndim_problem):
                if (i == self.rng_optimization.integers(self.ndim_problem)) or\
                        (self.rng_optimization.random() < p_cr[k]):
                    x_cr[k, i] = x_mu[k, i]
        return x_cr

    def select(self, args=None, x=None, y=None, x_cr=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                break
            yy = self._evaluate_fitness(x_cr[k], args)
            if yy < y[k]:
                x[k] = x_cr[k]
                y[k] = yy
        return x, y

    def iterate(self, args=None, x=None, y=None):
        x1_mu, x2_mu, x3_mu, f_p = self.mutate(x)
        x1_cr = self.bound(self.crossover(x1_mu, x, f_p[:, 0, 1]))
        x2_cr = self.bound(self.crossover(x2_mu, x, f_p[:, 1, 1]))
        x3_cr = self.bound(x3_mu)
        x, y = self.select(args, x, y, x1_cr)
        x, y = self.select(args, x, y, x2_cr)
        x, y = self.select(args, x, y, x3_cr)
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = DE.optimize(self, fitness_function)
        x, y = self.initialize(args)
        fitness.extend(y)
        while True:
            x, y = self.iterate(args, x, y)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results
