import numpy as np

from pypop7.optimizers.eda.eda import EDA
from scipy.linalg import orth


class RPEDA(EDA):
    """Random Projection Estimation of distribution algorithms(RP-EDA)
    Reference
    --------------
    A. Kaban, J. Bootkrajang, R. J. Durrant
    Towards Large Scale Continuous EDA: A Random Matrix Theory Perspective
    GECCO 2013: 383-390
    """
    def __init__(self, problem, options):
        EDA.__init__(self, problem, options)
        self.x = None
        self.k = options.get('k')
        self.typeR = options.get('typeR')
        self.rpmSize = options.get('rpmSize', 1000)
        self.sq_1_d = np.sqrt(1.0 / self.ndim_problem)
        self.sq_d_k = np.sqrt(self.ndim_problem / self.k)
        self.sq_rpm_size = np.sqrt(self.rpmSize)
        assert self.k < self.ndim_problem

    def initialize(self):
        x = np.empty((self.n_individuals, self.ndim_problem))
        x_fit = np.empty((self.n_parents, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            x[i] = self._initialize_x()
            y[i] = self._evaluate_fitness(x[i])
        return x, x_fit, y

    def iterate(self, x_fit, y):
        mean = np.mean(x_fit, axis=0)
        for i in range(self.n_parents):
            x_fit[i] -= mean
        x = np.zeros((self.n_individuals, self.ndim_problem))
        for i in range(self.rpmSize):
            if self.typeR == 'G':
                R = self.rng_optimization.standard_normal((self.ndim_problem, self.k)) * self.sq_1_d
            elif self.typeR == 'g':
                R = self.rng_optimization.standard_normal((self.ndim_problem, self.k)) * self.sq_1_d
            elif self.typeR == 'o':
                R = self.rng_optimization.standard_normal((self.ndim_problem, self.k)) * self.sq_1_d
                R = orth(R)
            elif self.typeR == 'b':
                R = self.rng_optimization.standard_normal((self.ndim_problem, self.k))
                for j in range(self.ndim_problem):
                    for k in range(self.k):
                        if R[j][k] > 0.5:
                            R[j][k] = self.sq_1_d
                        else:
                            R[j][k] = -1 * self.sq_1_d
            elif self.typeR == 's':
                R = self.rng_optimization.standard_normal((self.ndim_problem, self.k))
                for j in range(self.ndim_problem):
                    for k in range(self.k):
                        if R[j][k] > 5.0 / 6:
                            R[j][k] = -2
                        elif R[j][k] > 2.0 / 3:
                            R[j][k] = -1
                        elif R[j][k] > 0:
                            R[j][k] = 0
                        elif R[j][k] == -2:
                            R[j][k] = 1
                R *= np.sqrt(3) * self.sq_1_d
            else:
                raise Exception("Solver not implemented")
            project_matrix = np.dot(x_fit, R)
            cvr = np.zeros((self.k, self.k))
            for j in range(self.n_parents):
                cvr += np.cov(project_matrix[j])
            cvr /= self.n_parents
            xtmp = self.rng_optimization.multivariate_normal(np.zeros((self.k, )), cvr, self.n_individuals)
            x += np.dot(xtmp, np.transpose(R))
        x /= self.rpmSize
        x *= self.sq_d_k * self.sq_rpm_size
        for i in range(self.n_individuals):
            x[i] = np.clip(x[i] + mean, self.lower_boundary, self.upper_boundary)
            if self._check_terminations():
                return x, y
            y[i] = self._evaluate_fitness(x[i])
        return x, y

    def optimize(self, fitness_function=None):
        fitness = EDA.optimize(self, fitness_function)
        x, x_fit, y = self.initialize()
        while True:
            order = np.argsort(y)
            for i in range(self.n_parents):
                x_fit[i] = x[order[i]]
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            x, y = self.iterate(x_fit, y)
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results
