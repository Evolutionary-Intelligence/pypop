import numpy as np

from pypop7.optimizers.es.es import ES


class DES(ES):
    """Differential Evolution Strategy (DES).
    References
    ----------
    Arabas, J. and Jagodziński, D., 2020.
    Toward a matrix-free covariance matrix adaptation evolution strategy.
    IEEE Transactions on Evolutionary Computation, 24(1), pp.84-98.
    https://doi.org/10.1109/TEVC.2019.2907266
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.c_d = options.get('c_d', self.n_parents/(self.n_parents + 2.0))
        self.c_c = options.get('c_c', 1.0/np.sqrt(self.ndim_problem))
        self.c_epsilon = options.get('c_epsilon', 2.0/(self.ndim_problem**2))
        self.h = options.get('h', int(6 + 3*np.sqrt(self.ndim_problem)))
        self.epsilon = options.get('epsilon', 1)  # 1e-6
        self._c_d_1 = np.sqrt(self.c_d/2.0)
        self._c_d_2 = np.sqrt(self.c_d)
        self._c_d_3 = np.sqrt(1.0 - self.c_d)
        self._c_c_1 = np.sqrt(self.n_parents*self.c_c*(2.0 - self.c_c))

    def initialize(self, args=None):
        self._n_generations += 1
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # population
        y = np.empty((self.n_individuals,))  # fitness (cost)
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        order = np.argsort(y)[:self.n_parents]
        mean = np.vstack((np.mean(x, 0), np.mean(x[order], 0)))
        pp = mean[-1] - mean[-2]
        xx = np.copy(x[order])
        return x, y, mean, pp, xx

    def iterate(self, args=None, x=None, y=None, mean=None, pp=None, xx=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            tau = self.rng_optimization.choice(self.h, (3,)) + 1
            tau = np.minimum(self._n_generations, tau)
            if tau[0] != 1:
                x1, x2 = self.rng_optimization.choice(
                    xx[(-tau[0]*self.n_parents):((-tau[0]+1)*self.n_parents)], (2,), replace=False)
            else:
                x1, x2 = self.rng_optimization.choice(
                    xx[(-tau[0]*self.n_parents):], (2,), replace=False)
            d = (self._c_d_1*(x1 - x2) +
                 self._c_d_2*self.rng_optimization.standard_normal()*(mean[-tau[1]] - mean[-tau[1]-1]) +
                 self._c_d_3*self.rng_optimization.standard_normal()*pp[-tau[2]] +
                 self.epsilon*np.power(1.0 - self.c_epsilon, self._n_generations/2.0) *
                 self.rng_optimization.standard_normal((self.ndim_problem,)))
            x[k] = mean[-1] + d
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _update_distribution(self, x=None, y=None, mean=None, pp=None, xx=None):
        order = np.argsort(y)[:self.n_parents]
        mean = np.vstack((mean, np.mean(x[order], 0)))[-(self.h + 1):]
        p = (1 - self.c_c)*pp[-1] + self._c_c_1*(mean[-1] - mean[-2])
        pp = np.vstack((pp, p))[-self.h:]
        xx = np.vstack((xx, x[order]))[-self.h*self.n_parents:]
        return mean, pp, xx

    def optimize(self, fitness_function=None, args=None):
        fitness = ES.optimize(self, fitness_function)
        x, y, mean, pp, xx = self.initialize(args)
        fitness.extend(y)
        while True:
            x, y = self.iterate(args, x, y, mean, pp, xx)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, pp, xx = self._update_distribution(x, y, mean, pp, xx)
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness, mean[-1])
        return results
