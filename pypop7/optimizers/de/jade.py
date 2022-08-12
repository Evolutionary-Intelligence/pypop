import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class JADE(Optimizer):
    """Adaptive Differential Evolution (JADE).

    Reference
    ---------
    Zhang, J., and Sanderson, A. C. 2009.
    JADE: Adaptive differential evolution with optional external archive.
    IEEE Transactions on Evolutionary Computation, 13(5), 945–958.
    https://doi.org/10.1109/TEVC.2009.2014613
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self.mu_cr = 0.5
        self.mu_f = 0.5
        self.archive = np.empty((0, self.ndim_problem))
        self.p = 0.2
        self.c = 0.2  # 0.05 - 0.2
        self._n_generations = 0

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(
            self.initial_lower_boundary, self.initial_upper_boundary,
            (self.n_individuals, self.ndim_problem))  # initial point
        yy = []
        for i in range(self.n_individuals):
            yy.append(self._evaluate_fitness(x[i], args))
        return x, yy

    def iterate(self, args=None, x=None, fitness=None):
        cr = self.rng_optimization.normal(self.mu_cr, 0.1, (self.n_individuals,))
        f = self.rng_optimization.normal(self.mu_f, 0.1, (self.n_individuals,))

        # The union of the current population and the archive
        x_union = np.copy(x)
        if len(self.archive) > 0:
            x_union = np.vstack((x_union, self.archive))

        # r1 and r2
        n_ind = np.arange(self.n_individuals)
        r1 = np.array([self.rng_optimization.choice(np.setdiff1d(n_ind, np.array([i]))) for i in n_ind])
        r2 = np.array([self.rng_optimization.choice(
            np.setdiff1d(np.arange(len(x_union)), np.union1d(np.array([i]), r1[i]))) for i in n_ind])

        # top 100p%
        yy = fitness[-self.n_individuals:]
        sort_index = np.argsort(yy)[:int(self.p * self.n_individuals)]
        x_best = x[self.rng_optimization.choice(sort_index, (self.n_individuals,))]

        # Mutation
        f_m = np.tile(f.reshape(self.n_individuals, 1), (1, self.ndim_problem))
        v = x + f_m * (x_best - x) + f_m * (x[r1] - x_union[r2])

        # Crossover
        x_crossover = np.zeros((self.n_individuals, self.ndim_problem))
        r = self.rng_optimization.random((self.n_individuals, self.ndim_problem))
        for i in range(self.n_individuals):
            for j in range(self.ndim_problem):
                if (j == self.rng_optimization.integers(0, self.ndim_problem)) or (r[i][j] < cr[i]):
                    x_crossover[i, j] = v[i, j]
                else:
                    x_crossover[i, j] = x[i, j]

        # Selection
        s_f = []  # the set of all successful crossover probabilities
        s_cr = []  # the set of all successful mutation factors
        x_new = np.zeros((self.n_individuals, self.ndim_problem))
        for i in range(self.n_individuals):
            y_i = self._evaluate_fitness(x_crossover[i])
            if yy[i] <= y_i:
                x_new[i] = x[i]
                fitness.append(yy[i])
            else:
                x_new[i] = x_crossover[i]
                fitness.append(y_i)
                self.archive = np.vstack((self.archive, x_crossover[i]))
                s_cr.append(cr[i])
                s_f.append(f[i])

        # update
        self.mu_cr = (1 - self.c) * self.mu_cr + self.c * np.mean(np.array(s_cr))
        if np.sum(np.array(s_f)) != 0:
            self.mu_f = (1 - self.c) * self.mu_f + self.c * np.sum(np.power(np.array(s_f), 2)) / np.sum(np.array(s_f))
        else:
            self.mu_f = (1 - self.c) * self.mu_f
        return x_new, fitness

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        x, y = self.initialize(args)
        fitness.extend(y)
        while True:
            x, fitness = self.iterate(args, x, fitness)

            # Update archive
            if len(self.archive) > self.n_individuals:
                self.archive = self.archive[
                    self.rng_optimization.choice(np.arange(len(self.archive)), (self.n_individuals,))]

            if self._check_terminations():
                break
            self._n_generations += 1
        results = self._collect_results(fitness)
        return results
