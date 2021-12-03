import numpy as np

from optimizers.es.ssaes import SSAES


class DSAES(SSAES):
    """Derandomized Self-Adaptation Evolution Strategy (DSAES, Derandomized (1, λ)-σSA-ES).

    Reference
    ---------
    Hansen, N., Arnold, D.V. and Auger, A., 2015.
    Evolution strategies.
    In Springer Handbook of Computational Intelligence (pp. 871-898). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007%2F978-3-662-43505-2_44
    """
    def __init__(self, problem, options):
        if options.get('n_individuals') is None:
            options['n_individuals'] = 10  # mandatory setting for DSAES
        super(SSAES, self).__init__(problem, options)
        self.individual_sigmas = options.get('individual_sigmas')  # individual step-sizes
        # when only the global step size is given
        if (self.individual_sigmas is None) and (np.array(self.sigma).size == 1):
            self.individual_sigmas = self.sigma * np.ones((self.ndim_problem,))
        if self.eta_sigma is None:
            self.eta_sigma = 1 / 3
        # E[|N(0,1)|]: expectation of half-normal distribution
        self._e_hnd = np.sqrt(2 / np.pi)

    def iterate(self, x=None, mean=None, sigmas=None, y=None, args=None):
        for k in range(self.n_individuals):  # sample population
            if self._check_terminations():
                return x, sigmas, y
            sigma = self.eta_sigma * self.rng_optimization.standard_normal()
            z = self.rng_optimization.standard_normal((self.ndim_problem,))
            x[k] = mean + np.exp(sigma) * self.individual_sigmas * z
            sigma_1 = np.power(np.exp(np.abs(z) / self._e_hnd - 1), 1 / self.ndim_problem)
            sigma_2 = np.power(np.exp(sigma), 1 / np.sqrt(self.ndim_problem))
            sigmas[k] = self.individual_sigmas * sigma_1 * sigma_2
            y[k] = self._evaluate_fitness(x[k], args)
        return x, sigmas, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        super(SSAES, self).optimize(fitness_function)
        fitness = []  # store all fitness generated during evolution
        x, mean, sigmas, y = self.initialize()
        while True:
            # sample and evaluate offspring population
            x, sigmas, y = self.iterate(x, mean, sigmas, y, args)
            if self.record_fitness:
                fitness.extend(y.tolist())
            if self._check_terminations():
                break
            order = np.argsort(y)[0]
            self.individual_sigmas = np.copy(sigmas[order])
            mean = np.copy(x[order])
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        results['mean'] = mean
        results['individual_sigmas'] = self.individual_sigmas
        return results
