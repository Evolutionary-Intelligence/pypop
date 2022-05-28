import numpy as np

from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.dsaes import DSAES


class CSAES(DSAES):
    """Cumulative Step-size Adaptation Evolution Strategy (CSAES, (μ/μ,λ)-ES with search path).

    CSA: Cumulative Step-size Adaptation (a.k.a. cumulative path length control)

    Reference
    ---------
    Hansen, N., Arnold, D.V. and Auger, A., 2015.
    Evolution strategies.
    In Springer Handbook of Computational Intelligence (pp. 871-898). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007%2F978-3-662-43505-2_44
    (See Algorithm 44.6 for details.)

    Ostermeier, A., Gawelczyk, A. and Hansen, N., 1994, October.
    Step-size adaptation based on non-local use of selection information
    In International Conference on Parallel Problem Solving from Nature (pp. 189-198).
    Springer, Berlin, Heidelberg.
    http://link.springer.com/chapter/10.1007/3-540-58484-6_263
    """
    def __init__(self, problem, options):
        if options.get('n_individuals') is None:
            options['n_individuals'] = 4 + int(np.floor(3 * np.log(problem.get('ndim_problem'))))
        if options.get('n_parents') is None:
            options['n_parents'] = int(options['n_individuals'] / 4)
        if options.get('eta_sigma') is None:
            options['eta_sigma'] = np.sqrt(options['n_parents'] / (problem['ndim_problem'] + options['n_parents']))
        DSAES.__init__(self, problem, options)
        self._s_1 = None
        self._s_2 = None
        # E[||N(0,I)||]: expectation of chi distribution
        self._e_chi = np.sqrt(self.ndim_problem) * (
                1 - 1 / (4 * self.ndim_problem) + 1 / (21 * np.power(self.ndim_problem, 2)))

    def initialize(self, is_restart=False):
        self.n_parents = int(self.n_individuals / 4)
        self.eta_sigma = np.sqrt(self.n_parents / (self.ndim_problem + self.n_parents))
        self._s_1 = 1 - self.eta_sigma
        self._s_2 = np.sqrt(self.eta_sigma * (2 - self.eta_sigma) * self.n_parents)
        self._axis_sigmas = self.sigma * np.ones((self.ndim_problem,))
        z = np.empty((self.n_individuals, self.ndim_problem))  # noise for offspring population
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        s = np.zeros((self.ndim_problem,))  # evolution path
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return z, x, mean, s, y

    def iterate(self, z=None, x=None, mean=None, y=None, args=None):
        for k in range(self.n_individuals):  # sample population (Line 4)
            if self._check_terminations():
                return z, x, y
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))  # Line 5
            x[k] = mean + self._axis_sigmas * z[k]  # Line 6
            y[k] = self._evaluate_fitness(x[k], args)
        return z, x, y

    def _update_distribution(self, z=None, x=None, s=None, y=None):
        order = np.argsort(y)[:self.n_parents]  # Line 7
        s = self._s_1 * s + self._s_2 * np.mean(z[order], axis=0)  # Line 8
        sigmas_1 = np.power(np.exp(np.abs(s) / self._e_hnd - 1), 1 / (3 * self.ndim_problem))
        sigmas_2 = np.power(np.exp(np.linalg.norm(s) / self._e_chi - 1),
                            self.eta_sigma / (1 + np.sqrt(self.n_parents / self.ndim_problem)))
        self._axis_sigmas *= (sigmas_1 * sigmas_2)  # Line 9
        mean = np.mean(x[order], axis=0)  # Line 11
        return s, mean

    def restart_initialize(self, z=None, x=None, mean=None, s=None, y=None):
        is_restart = self._restart_initialize()
        if is_restart:
            z, x, mean, s, y = self.initialize(True)
        return z, x, mean, s, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        z, x, mean, s, y = self.initialize()
        while True:
            # sample and evaluate offspring population
            z, x, y = self.iterate(z, x, mean, y, args)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            s, mean = self._update_distribution(z, x, s, y)
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                z, x, mean, s, y = self.restart_initialize(z, x, mean, s, y)
        results = self._collect_results(fitness, mean)
        results['s'] = s
        results['_axis_sigmas'] = self._axis_sigmas
        return results
