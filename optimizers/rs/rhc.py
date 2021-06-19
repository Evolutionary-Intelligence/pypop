import numpy as np

from optimizers.rs.rs import RS


class RHC(RS):
    """Random Hill Climber (RHC).

    Only support normally distributed random sampling during optimization.
    But support uniformly or normally distributed random sampling for the initial starting point.

    Reference
    ---------
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/hillclimber.py
    """
    def __init__(self, problem, options):
        RS.__init__(self, problem, options)
        self.initialization_distribution = options.get('initialization_distribution', 1)
        if self.initialization_distribution not in [0, 1]:  # 0 -> normally distributed
            raise ValueError('Only support uniformly (1) or normally (0) distributed random initialization.')
        self.initial_std = options.get('initial_std', 1.0)
        self.global_std = options.get('global_std', 0.1)

    def _sample(self, rng):
        if self.initialization_distribution == 0:
            x = rng.standard_normal(size=(self.ndim_problem,)) * self.initial_std
        else:
            x = rng.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        return x

    def initialize(self):
        if self.x is None:
            x = self._sample(self.rng_initialization)
        else:
            x = np.copy(self.x)
        return x

    def iterate(self):
        # mutate the best-so-far individual via adding Gaussian noise
        mutation = self.rng_optimization.standard_normal(size=(self.ndim_problem,))
        return self.best_so_far_x + self.global_std * mutation
