import numpy as np

from optimizers.core.optimizer import Optimizer


class NES(Optimizer):
    def __init__(self, problem, options):
        problem['_is_maximization'] = True
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # population size
            self.n_individuals = 4 + int(np.floor(3 * np.log(self.ndim_problem)))
        self.eta_mu = options.get('eta_mu')  # learning rate of mean of Gaussian search distribution
        self.eta_sigma = options.get('eta_sigma')  # learning rate of std of Gaussian search distribution
        self.mu = options.get('mu')
        self.n_generations = options.get('n_generations', 0)

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def optimize(self, fitness_function=None):
        raise NotImplementedError

    def _initialize_mu(self):
        if self.mu is None:
            rng = self.rng_initialization
            mu = rng.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        else:
            mu = np.copy(self.mu)
        return mu

    def _compute_gradients(self):
        raise NotImplementedError

    def _update_distribution(self):
        raise NotImplementedError
