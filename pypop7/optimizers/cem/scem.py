import numpy as np

from pypop7.optimizers.cem.cem import CEM


class SCEM(CEM):
    """Standard Cross-Entropy Method (SCEM).

    References
    ----------
    Kroese, D.P., Porotsky, S. and Rubinstein, R.Y., 2006.
    The cross-entropy method for continuous multi-extremal optimization.
    Methodology and Computing in Applied Probability, 8(3), pp.383-407.
    https://link.springer.com/article/10.1007/s11009-006-9753-0
    (See [B Main CE Program] for the official Matlab code.)

    De Boer, P.T., Kroese, D.P., Mannor, S. and Rubinstein, R.Y., 2005.
    A tutorial on the cross-entropy method.
    Annals of Operations Research, 134(1), pp.19-67.
    https://link.springer.com/article/10.1007/s10479-005-5724-z
    """
    def __init__(self, problem, options):
        CEM.__init__(self, problem, options)
        self.alpha = options.get('alpha', 0.8)  # smoothing factor

    def initialize(self, is_restart=False):
        mean = self._initialize_mean(is_restart)
        x = np.empty((self.n_individuals, self.ndim_problem))  # samples (population)
        y = np.empty((self.n_individuals,))  # fitness (cost)
        return mean, x, y

    def iterate(self, mean=None, x=None, y=None, args=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            x[i] = mean + self._sigmas*self.rng_optimization.standard_normal((self.ndim_problem,))
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y

    def _update_parameters(self, mean=None, x=None, y=None):
        xx = x[np.argsort(y)[:self.n_parents]]
        mean = self.alpha*np.mean(xx, axis=0) + (1.0-self.alpha)*mean
        self._sigmas = self.alpha*np.std(xx, axis=0) + (1.0-self.alpha)*self._sigmas
        return mean

    def optimize(self, fitness_function=None, args=None):
        fitness = CEM.optimize(self, fitness_function)
        mean, x, y = self.initialize()
        while True:
            x, y = self.iterate(mean, x, y, args)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            mean = self._update_parameters(mean, x, y)
        return self._collect_results(fitness, mean)
