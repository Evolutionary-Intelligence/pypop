import numpy as np

from optimizers.rs.rs import RS


class CSA(RS):
    """Simulated Annealing (SA) designed by Corana et al., ACM-TOMS, 1987.

    Reference
    ---------
    Kirkpatrick, S., Gelatt, C.D. and Vecchi, M.P., 1983.
    Optimization by simulated annealing.
    Science, 220(4598), pp.671-680.
    https://science.sciencemag.org/content/220/4598/671

    Corana, A., Marchesi, M., Martini, C. and Ridella, S., 1987.
    Minimizing multimodal functions of continuous variables with the “simulated annealing” algorithm.
    ACM Transactions on Mathematical Software, 13(3), pp.262-280.
    https://dl.acm.org/doi/abs/10.1145/29380.29864
    https://dl.acm.org/doi/10.1145/66888.356281

    https://esa.github.io/pygmo2/algorithms.html#pygmo.simulated_annealing
    """
    def __init__(self, problem, options):
        RS.__init__(self, problem, options)
        step_vector = (self.initial_upper_boundary - self.initial_lower_boundary) * 0.01
        self.v = options.get('v', step_vector)  # starting step vector
        self.T = options.get('T', 10.0)  # starting temperature
        self.N_S = options.get('N_S', 10)  # for step variation
        self.c = options.get('c', 2 * np.ones(self.ndim_problem,))  # for step variation criterion
        self.N_T = options.get('N_T', 10)  # for temperature reduction
        self.r_T = options.get('r_T', 0.85)  # for temperature reduction coefficient
        self.n = np.zeros((self.ndim_problem,))  # for step variation
        self.parent_x = np.copy(self.best_so_far_x)
        self.parent_y = np.copy(self.best_so_far_y)

    def initialize(self, args=None):
        if self.x is None:  # starting point
            x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        else:
            x = np.copy(self.x)
        y = self._evaluate_fitness(x, args)
        self.parent_x, self.parent_y = x, y
        return y

    def iterate(self, args=None):  # perform a cycle of random moves
        fitness = []
        for h in range(self.ndim_problem):
            x = np.copy(self.parent_x)
            x[h] = self.parent_x[h] + self.rng_optimization.uniform(-1, 1) * self.v[h]
            y = self._evaluate_fitness(x, args)
            if self.record_fitness:
                fitness.append(y)
            self._print_verbose_info()
            diff = self.parent_y - y
            if (diff >= 0) or (self.rng_optimization.random() < np.exp(diff / self.T)):
                self.parent_x, self.parent_y = x, y
                self.n[h] += 1
            if self._check_terminations():
                break
        return fitness

    def adjust_step_vector(self):
        for u in range(self.ndim_problem):
            if self.n[u] > 0.6 * self.N_S:
                self.v[u] *= 1 + self.c[u] * (self.n[u] / self.N_S - 0.6) / 0.4
            elif self.n[u] < 0.4 * self.N_S:
                self.v[u] /= 1 + self.c[u] * (0.4 - self.n[u] / self.N_S) / 0.4
        self.n = np.zeros((self.ndim_problem,))

    def _collect_results(self, fitness):
        results = RS._collect_results(self, fitness)
        results['v'] = np.copy(self.v)
        return results

    def optimize(self, fitness_function=None, args=None):
        super(RS, self).optimize(fitness_function)
        fitness = [self.initialize(args)]  # store all fitness generated during search
        while not self._check_terminations():
            for m in range(self.N_T):
                for j in range(self.N_S):
                    fitness.extend(self.iterate(args))
                    if self._check_terminations():
                        break
                if self._check_terminations():
                    break
                self.adjust_step_vector()
            self.T *= self.r_T  # reduce temperature
            self.parent_x, self.parent_y = self.best_so_far_x, self.best_so_far_y
        return self._collect_results(fitness)
