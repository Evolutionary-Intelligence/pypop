import numpy as np

from optimizers.es.es import ES


class SSAES(ES):
    """Schwefel's Self-Adaptation Evolution Strategy (SSAES, (μ/μ, λ)-σSA-ES).

    Reference
    ---------
    Hansen, N., Arnold, D.V. and Auger, A., 2015.
    Evolution strategies.
    In Springer Handbook of Computational Intelligence (pp. 871-898). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007%2F978-3-662-43505-2_44
    """
    def __init__(self, problem, options):
        if options.get('n_individuals') is None:
            options['n_individuals'] = 5 * problem.get('ndim_problem')  # mandatory setting for SSAES
        if options.get('n_parents') is None:
            options['n_parents'] = int(options['n_individuals'] / 4)  # mandatory setting for SSAES
        ES.__init__(self, problem, options)
        self.individual_sigmas = options.get('individual_sigmas')  # individual step-sizes
        # when only the global step size is given
        if (self.individual_sigmas is None) and (np.array(self.sigma).size == 1):
            self.individual_sigmas = self.sigma * np.ones((self.ndim_problem,))
        if self.eta_sigma is None:
            self.eta_sigma = 1 / np.sqrt(self.ndim_problem)
        # learning rate for individual step-sizes
        self.eta_individual_sigmas = options.get('eta_individual_sigmas', 1 / np.power(self.ndim_problem, 1 / 4))

    def initialize(self):
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean()  # mean of Gaussian search distribution
        sigmas = np.ones((self.n_individuals, self.ndim_problem))  # individual step-sizes for all offspring
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return x, mean, sigmas, y

    def iterate(self, x=None, mean=None, sigmas=None, y=None, args=None):
        for k in range(self.n_individuals):  # sample population
            if self._check_terminations():
                return x, sigmas, y
            sigma = self.eta_sigma * self.rng_optimization.standard_normal()
            coordinate_sigmas = self.eta_individual_sigmas * self.rng_optimization.standard_normal((self.ndim_problem,))
            sigmas[k] = self.individual_sigmas * np.exp(coordinate_sigmas) * np.exp(sigma)
            x[k] = mean + sigmas[k] * self.rng_optimization.standard_normal((self.ndim_problem,))
            y[k] = self._evaluate_fitness(x[k], args)
        return x, sigmas, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        ES.optimize(self, fitness_function)
        fitness = []  # store all fitness generated during evolution
        x, mean, sigmas, y = self.initialize()
        while True:
            # sample and evaluate offspring population
            x, sigmas, y = self.iterate(x, mean, sigmas, y, args)
            if self.record_fitness:
                fitness.extend(y.tolist())
            if self._check_terminations():
                break
            order = np.argsort(y)[:self.n_parents]
            self.individual_sigmas = np.mean(sigmas[order], axis=0)
            mean = np.mean(x[order], axis=0)
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        results['mean'] = mean
        results['individual_sigmas'] = self.individual_sigmas
        return results
