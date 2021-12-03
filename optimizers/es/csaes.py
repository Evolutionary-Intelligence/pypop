import numpy as np

from optimizers.es.es import ES


class CSAES(ES):
    """Cumulative Step-size Adaptation Evolution Strategy (CSAES, (μ/μ, λ)-ES with search path).

    CSA: Cumulative Step-size Adaptation (a.k.a. cumulative path length control)

    Reference
    ---------
    Hansen, N., Arnold, D.V. and Auger, A., 2015.
    Evolution strategies.
    In Springer Handbook of Computational Intelligence (pp. 871-898). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007%2F978-3-662-43505-2_44
    """
    def __init__(self, problem, options):
        if options.get('n_individuals') is None:
            options['n_individuals'] = 4 + int(np.floor(3 * np.log(problem.get('ndim_problem'))))
        if options.get('n_parents') is None:
            options['n_parents'] = int(options['n_individuals'] / 4)
        ES.__init__(self, problem, options)
        if self.eta_sigma is None:
            self.eta_sigma = np.sqrt(self.n_parents / (self.ndim_problem + self.n_parents))
        self._s_1 = 1 - self.eta_sigma
        self._s_2 = np.sqrt(self.eta_sigma * (2 - self.eta_sigma) * self.n_parents)
        self.individual_sigmas = options.get('individual_sigmas')  # individual step-sizes
        # when only the global step size is given
        if (self.individual_sigmas is None) and (np.array(self.sigma).size == 1):
            self.individual_sigmas = self.sigma * np.ones((self.ndim_problem,))
        # E[|N(0,1)|]: expectation of half-normal distribution
        self._e_hnd = np.sqrt(2 / np.pi)
        # E[||N(0,I)||]: expectation of chi distribution
        self._e_chi = np.sqrt(self.ndim_problem) * (
                1 - 1 / (4 * self.ndim_problem) + 1 / (21 * np.power(self.ndim_problem, 2)))

    def initialize(self):
        z = np.empty((self.n_individuals, self.ndim_problem))  # noise for offspring population
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean()  # mean of Gaussian search distribution
        s = np.zeros((self.ndim_problem,))  # evolution path
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return z, x, mean, s, y

    def iterate(self, z=None, x=None, mean=None, y=None, args=None):
        for k in range(self.n_individuals):  # sample population
            if self._check_terminations():
                return z, x, y
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            x[k] = mean + self.individual_sigmas * z[k]
            y[k] = self._evaluate_fitness(x[k], args)
        return z, x, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        ES.optimize(self, fitness_function)
        fitness = []  # store all fitness generated during evolution
        z, x, mean, s, y = self.initialize()
        while True:
            # sample and evaluate offspring population
            z, x, y = self.iterate(z, x, mean, y, args)
            if self.record_fitness:
                fitness.extend(y.tolist())
            if self._check_terminations():
                break
            order = np.argsort(y)[:self.n_parents]
            s = self._s_1 * s + self._s_2 * np.mean(z[order], axis=0)
            sigma_1 = np.power(np.exp(np.abs(s) / self._e_hnd - 1), 1 / (3 * self.ndim_problem))
            sigma_2 = np.power(np.exp(np.linalg.norm(s) / self._e_chi - 1),
                               self.eta_sigma / (1 + np.sqrt(self.n_parents / self.ndim_problem)))
            self.individual_sigmas *= sigma_1 * sigma_2
            mean = np.mean(x[order], axis=0)
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        results['mean'] = mean
        results['s'] = s
        results['individual_sigmas'] = self.individual_sigmas
        return results
