import numpy as np

from optimizers.core.optimizer import Optimizer


class NES(Optimizer):
    """Natural Evolution Strategies (NES).

    Reference
    ---------
    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014.
    Natural evolution strategies.
    Journal of Machine Learning Research, 15(27), pp.949-980.
    https://jmlr.org/papers/v15/wierstra14a.html
    """
    def __init__(self, problem, options):
        problem['_is_maximization'] = True  # mandatory setting for NES
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # population size (λ: lambda)
            self.n_individuals = 4 + int(np.floor(3 * np.log(self.ndim_problem)))
        self.mu = options.get('mu')  # mean of Gaussian search distribution (μ)
        self.eta_mu = options.get('eta_mu')  # learning rate of mean (η_μ)
        self.eta_sigma = options.get('eta_sigma')  # learning rate of std (η_σ)
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

    def _print_verbose_info(self, y):
        if self.verbose and (not self.n_generations % self.verbose_frequency):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self.n_generations, -self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect_results(self):
        results = Optimizer._collect_results(self)
        results['n_generations'] = self.n_generations
        return results

    def _fitness_shaping(self):
        base = np.log(self.n_individuals / 2 + 1)
        utilities = np.array([max(0, base - np.log(k)) for k in (np.arange(self.n_individuals) + 1)])
        return utilities / np.sum(utilities) - (1 / self.n_individuals)
