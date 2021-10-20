import numpy as np

from optimizers.es.es import ES


class SAES(ES):
    """Self-Adaptation Evolution Strategy (SAES, (μ/μ_I, λ)-σSA-ES)

    Reference
    ---------
    Beyer, H.G., 2020, July.
    Design principles for matrix adaptation evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation Companion (pp. 682-700).
    https://dl.acm.org/doi/abs/10.1145/3377929.3389870

    http://www.scholarpedia.org/article/Evolution_strategies

    https://homepages.fhv.at/hgb/downloads/mu_mu_I_lambda-ES.oct    (see the official Matlab/Octave version)
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        if self.eta_sigma is None:
            self.eta_sigma = 1 / np.sqrt(2 * self.ndim_problem)

    def initialize(self):
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mu = self._initialize_mu()  # mean of Gaussian search distribution
        sigmas = np.ones((self.n_individuals,))  # step-sizes for each offspring
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return x, mu, sigmas, y

    def iterate(self, x=None, mu=None, sigmas=None, y=None, args=None):
        for k in range(self.n_individuals):  # sample population
            if self._check_terminations():
                return x, sigmas, y
            sigmas[k] = self.sigma * np.exp(self.eta_sigma * self.rng_optimization.standard_normal())
            x[k] = mu + sigmas[k] * self.rng_optimization.standard_normal((self.ndim_problem,))
            y[k] = self._evaluate_fitness(x[k], args)
        return x, sigmas, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        ES.optimize(self, fitness_function)
        fitness = []  # store all fitness generated during evolution
        x, mu, sigmas, y = self.initialize()
        while True:
            x, sigmas, y = self.iterate(x, mu, sigmas, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y.tolist())
            if self._check_terminations():
                break
            order = np.argsort(y)[:self.n_parents]
            mu = np.mean(x[order], axis=0)
            self.sigma = np.mean(sigmas[order])
            self.n_generations += 1
            self._print_verbose_info(y)
        if self.record_fitness:
            self._compress_fitness(fitness[:self.n_function_evaluations])
        results = self._collect_results()
        results['mu'] = mu
        return results
