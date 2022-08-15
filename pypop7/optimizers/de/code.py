import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class CODE(Optimizer):
    """Composite Differential Evolution (CoDE).

    Reference
    ---------
    Wang, Y., Cai, Z., and Zhang, Q. 2011.
    Differential evolution with composite trial vector generation strategies and control parameters.
    IEEE Transactions on Evolutionary Computation, 15(1), 55–66.
    https://doi.org/10.1109/TEVC.2010.2087271
    """

    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self._n_generations = 0
        self.c_p = [[1.0, 0.1], [1.0, 0.9], [0.8, 0.2]]  # [f, cr] control parameter settings pool
        self.n_individuals = options.get('n_individuals', 30)

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(
            self.initial_lower_boundary, self.initial_upper_boundary,
            (self.n_individuals, self.ndim_problem))  # initial point
        y = []
        for i in range(self.n_individuals):
            y.append(self._evaluate_fitness(x[i], args))
        return x, y

    def mutation(self, x, xx=None):
        # Mutation
        c = self.rng_optimization.choice([0, 1, 2], (3,))  # randomly selected from the parameter candidate pool
        p = self.rng_optimization.permutation(self.n_individuals)
        v_1 = xx[p[0]] + self.c_p[c[0]][0] * (xx[p[1]] - xx[p[2]])  # rand/1/bin

        p = self.rng_optimization.permutation(self.n_individuals)
        v_2 = xx[p[0]] + self.c_p[c[1]][0] * (xx[p[1]] - xx[p[2]]) + \
              self.c_p[c[1]][0] * (xx[p[3]] - xx[p[4]])  # rand/2/bin

        p = self.rng_optimization.permutation(self.n_individuals)
        v_3 = x + self.rng_optimization.random() * (xx[p[0]] - x) + \
              self.c_p[c[2]][0] * (xx[p[1]] - xx[p[2]])  # current-to-rand/1
        return [v_1, v_2, v_3], c

    def crossover(self, v, x, cr):
        u = np.zeros((self.ndim_problem,))
        for j in range(self.ndim_problem):
            if (j == self.rng_optimization.integers(0, self.ndim_problem)) or \
                    (self.rng_optimization.random() < cr):
                u[j] = v[j]
            else:
                u[j] = x[j]
        return u

    def iterate(self, args=None, x=None, fitness=None):
        u = np.zeros((self.n_individuals, self.ndim_problem))
        y_u = []
        for i in range(self.n_individuals):
            # mutation
            v, c = self.mutation(x[i], x)
            for j in [0, 1]:
                v[j] = self.crossover(v[j], x[i], self.c_p[c[j]][1])

            for j in [0, 1, 2]:
                b = v[j] < self.lower_boundary  # bool values
                v[j][b] = np.minimum(self.upper_boundary, 2 * self.lower_boundary - v[j])[b]
                b = v[j] > self.upper_boundary  # bool values
                v[j][b] = np.maximum(self.lower_boundary, 2 * self.upper_boundary - v[j])[b]

            # choose the best trial vector
            y_v = [self._evaluate_fitness(v[j], args) for j in range(len(v))]
            idx = np.argsort(y_v)[0]
            u[i] = v[idx]
            y_u.append(y_v[idx])

        # Selection
        x_new = np.copy(x)
        y_new = fitness[-self.n_individuals:]
        b = y_u < y_new  # bool values
        x_new[b] = u[b]
        y_new[b] = y_u[b]
        fitness.extend(y_new)
        return x_new, fitness

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        x, y = self.initialize(args)
        fitness.extend(y)
        while True:
            x, fitness = self.iterate(args, x, fitness)

            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
        results = self._collect_results(fitness)
        return results
