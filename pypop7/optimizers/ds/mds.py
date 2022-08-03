import numpy as np

from pypop7.optimizers.ds.ds import DS


class MDS(DS):
    """Multi-directional Search(MDS)
        Reference
        ------------
        V. Torczon
        ON THE CONVERGENCE OF THE MULTIDIRECTIONAL SEARCH ALGORITHM
        SIAM J. OPTIMIZATION Vol.1, No. 1, pp. 123-145, February 1991
        https://doi.org/10.1137/0801010
        V. Torczon
        Multidirectional search: a direct search algorithm for parallel machines
        Computer Science 1989
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.9967&rep=rep1&type=pdf
    """
    def __init__(self, problem, options):
        DS.__init__(self, problem, options)
        self.mu = options.get('mu', 2.5)
        self.theta = options.get('theta', 0.25)
        self.x = None

    def initialize(self, is_restart=False):
        x = np.empty((self.n_individuals + 1, self.ndim_problem))
        y = np.empty((self.n_individuals + 1,))
        for i in range(self.n_individuals + 1):
            x[i] = self._initialize_x(is_restart)
            y[i] = self._evaluate_fitness(x[i])
        return x, y

    def iterate(self, x, y, fitness):
        r = np.empty((self.n_individuals, self.ndim_problem))
        f_r = np.empty((self.n_individuals,))
        place = np.argmin(y)
        # swap min to place 0
        tempx, tempy = x[place], y[place]
        x[place], y[place] = x[0], y[0]
        x[0], y[0] = tempx, tempy
        while True:
            if self._check_terminations():
                break
            for i in range(self.n_individuals):
                r[i] = 2 * x[0] - x[i+1]
                f_r[i] = self._evaluate_fitness(r[i])
            replaced = min(f_r) < y[0]
            if replaced:
                # expansion step
                e = np.empty((self.n_individuals, self.ndim_problem))
                f_e = np.empty((self.n_individuals,))
                for i in range(self.n_individuals):
                    e[i] = (1 - self.mu) * x[0] + self.mu * x[i]
                    f_e[i] = self._evaluate_fitness(e[i])
                if min(f_e) < min(f_r):
                    # accept expansion
                    for i in range(self.n_individuals):
                        x[i+1] = e[i]
                        y[i+1] = f_e[i]
                else:
                    # accept rotation
                    for i in range(self.n_individuals):
                        x[i+1] = r[i]
                        y[i+1] = f_r[i]
            else:
                # contraction step
                c = np.empty((self.n_individuals, self.ndim_problem))
                f_c = np.empty((self.n_individuals,))
                for i in range(self.n_individuals):
                    c[i] = (1 - self.theta) * x[0] + self.theta * x[i+1]
                    f_c[i] = self._evaluate_fitness(c[i])
                replaced = min(f_c) < y[0]
                # accept contraction
                for i in range(self.n_individuals):
                    x[i+1] = c[i]
                    y[i+1] = f_c[i]
            if self.record_fitness:
                fitness.append(y)
            if replaced:
                break
        return x, y, fitness

    def optimize(self, fitness_function=None):
        fitness = DS.optimize(self, fitness_function)
        x, y = self.initialize()
        fitness.append(y)
        while True:
            if self._check_terminations():
                break
            x, y, fitness = self.iterate(x, y, fitness)
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results
