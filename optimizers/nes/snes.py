import time
import numpy as np

from optimizers.nes.nes import NES
from optimizers.nes.utils import fitness_shaping


class SNES(NES):
    def __init__(self, problem, options):
        NES.__init__(self, problem, options)
        if self.eta_mu is None:  # learning rate of mean of Gaussian search distribution
            self.eta_mu = 1
        if self.eta_sigma is None:  # learning rate of std of Gaussian search distribution
            self.eta_sigma = (3 + np.log(self.ndim_problem)) / (5 * np.sqrt(self.ndim_problem))

    def initialize(self):
        s = np.empty((self.n_individuals, self.ndim_problem))  # sample
        mu = self._initialize_mu()  # mean of Gaussian search distribution
        sigma = np.ones((self.ndim_problem,))  # individual step-sizes (std of Gaussian search distribution)
        y = np.empty((self.n_individuals,))  # fitness
        return s, mu, sigma, y

    def iterate(self, s=None, mu=None, sigma=None, y=None):
        for k in range(self.n_individuals):
            termination_signal = self._check_terminations()
            if termination_signal[0]:
                return s, y
            # draw sample
            s[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            z = mu + sigma * s[k]
            # evaluate fitness
            self.start_function_evaluations = time.time()
            y[k] = self.fitness_function(z)
            self.n_function_evaluations += 1
            self.time_function_evaluations += time.time() - self.start_function_evaluations
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
        if fitness_function is not None:
            self.fitness_function = fitness_function
        s, mu, sigma, y = self.initialize()
        while True:
            # sample and evaluate offspring in batch
            s, y = self.iterate(s, mu, sigma, y)
            termination_signal = self._check_terminations()
            if termination_signal[0]:
                self.termination_signal = termination_signal[1]
                break
            g_mu, g_sigma = self._compute_gradients(s, y, fitness_shaping)
            mu, sigma = self._update_distribution(mu, sigma, g_mu, g_sigma)
            self.n_generations += 1
            print('  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e}'.format(
                self.n_generations, -self.best_so_far_y, np.min(y)))
        results = self._collect_results()
        results['mu'] = mu
        results['sigma'] = sigma
        return results
