import time
import numpy as np

from optimizers.nes.nes import NES
from optimizers.nes.utils import fitness_shaping


class SNES(NES):
    """Separable Natural Evolution Strategy (SNES).

    Reference
    ---------
    Schaul, T., Glasmachers, T. and Schmidhuber, J., 2011, July.
    High dimensions and heavy tails for natural evolution strategies.
    In Proceedings of the Annual Conference on Genetic and Evolutionary Computation (pp. 845-852).
    https://dl.acm.org/doi/abs/10.1145/2001576.2001692
    """
    def __init__(self, problem, options):
        NES.__init__(self, problem, options)
        if self.eta_mu is None:  # learning rate of mean of Gaussian search distribution
            self.eta_mu = 1
        if self.eta_sigma is None:  # learning rate of std of Gaussian search distribution
            self.eta_sigma = (3 + np.log(self.ndim_problem)) / (5 * np.sqrt(self.ndim_problem))

    def initialize(self):
        s = np.empty((self.n_individuals, self.ndim_problem))  # samples (population)
        mu = self._initialize_mu()  # mean of Gaussian search distribution
        sigma = np.ones((self.ndim_problem,))  # individual step-sizes (std of Gaussian search distribution)
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return s, mu, sigma, y

    def iterate(self, s=None, mu=None, sigma=None, y=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return s, y
            # draw sample (individual)
            s[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            z = mu + sigma * s[k]
            # evaluate fitness
            self.start_function_evaluations = time.time()
            y[k] = self.fitness_function(z)
            self.time_function_evaluations += time.time() - self.start_function_evaluations
            self.n_function_evaluations += 1
            # update best-so-far solution and fitness
            if -y[k] > self.best_so_far_y:  # maximization
                self.best_so_far_y = -y[k]
                self.best_so_far_x = np.copy(z)
        return s, y

    def _compute_gradients(self, s=None, y=None, upsilon_func=None):
        order = np.argsort(y)
        upsilon = upsilon_func(np.arange(y.size))
        upsilon[order] = np.copy(upsilon)
        g_mu = np.dot(upsilon, s)
        g_sigma = np.dot(upsilon, np.power(s, 2) - 1)
        return g_mu, g_sigma

    def _update_distribution(self, mu=None, sigma=None, g_mu=None, g_sigma=None):
        mu += self.eta_mu * sigma * g_mu  # maximization
        sigma *= np.exp(self.eta_sigma / 2 * g_sigma)  # maximization
        return mu, sigma

    def optimize(self, fitness_function=None):
        self.start_time = time.time()
        fitness = []  # store all fitness generated during evolution
        if fitness_function is not None:
            self.fitness_function = fitness_function
        s, mu, sigma, y = self.initialize()
        while True:
            # sample and evaluate offspring population
            s, y = self.iterate(s, mu, sigma, y)
            if self.record_options['record_fitness']:
                fitness.extend(y.tolist())
            if self._check_terminations():
                break
            g_mu, g_sigma = self._compute_gradients(s, y, fitness_shaping)
            mu, sigma = self._update_distribution(mu, sigma, g_mu, g_sigma)
            self.n_generations += 1
            self._print_verbose_info(y)
        if self.record_options['record_fitness']:
            self._compress_fitness(fitness[:self.n_function_evaluations])
        results = self._collect_results()
        results['mu'] = mu
        results['sigma'] = sigma
        return results
