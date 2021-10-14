import numpy as np

from optimizers.core.optimizer import Optimizer


class ES(Optimizer):
    """Evolution Strategies (ES).

    Reference
    ---------
    http://www.scholarpedia.org/article/Evolution_strategies

    Beyer, H.G. and Schwefel, H.P., 2002.
    Evolution strategies–A comprehensive introduction.
    Natural Computing, 1(1), pp.3-52.
    https://link.springer.com/article/10.1023/A:1015059928466

    Schwefel, H.P., 1984.
    Evolution strategies: A family of non-linear optimization techniques based on
        imitating some principles of organic evolution.
    Annals of Operations Research, 1(2), pp.165-167.
    https://link.springer.com/article/10.1007/BF01876146
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # offspring population size (λ: lambda)
            self.n_individuals = 4 + int(np.floor(3 * np.log(self.ndim_problem)))
        # to avoid notation confusion in μ, use `n_parents` and `mu` to represent
        #   parent population size and mean of Gaussian search distribution, respectively
        if self.n_parents is None:  # parent population size (μ: mu)
            self.n_parents = int(self.n_individuals / 2)
            w_base, w = np.log((self.n_individuals + 1) / 2), np.log(np.arange(self.n_parents) + 1)
            self.w = (w_base - w) / (self.n_parents * w_base - np.sum(w))
            self.mu_eff = 1 / np.sum(np.power(self.w, 2))  # μ_eff
        self.mu = options.get('mu')  # mean of Gaussian search distribution (μ)
        if self.mu is None:  # 'mu' has priority over 'x'
            self.mu = options.get('x')
        self.sigma = options.get('sigma', 0.1)  # global step-size (σ)
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

    def _print_verbose_info(self, y):
        if self.verbose and (not self.n_generations % self.verbose_frequency):
            best_so_far_y = -self.best_so_far_y if self._is_maximization else self.best_so_far_y
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self.n_generations, best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect_results(self):
        results = Optimizer._collect_results(self)
        results['mu'] = self.mu
        results['sigma'] = self.sigma
        results['n_generations'] = self.n_generations
        return results
