import numpy as np

from optimizers.rs.rs import RS


class RHC(RS):
    """Random (Stochastic) Hill Climber (RHC).

    Reference
    ---------
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/hillclimber.py
    """
    def __init__(self, problem, options):
        RS.__init__(self, problem, options)
        self.initial_std = options.get('initial_std', 1.0)
        self.global_std = options.get('global_std', 0.1)

    def initialize(self):
        if self.x is None:
            x = self.rng_initialization.standard_normal(size=(self.ndim_problem,))
            x *= self.initial_std
        else:
            x = np.copy(self.x)
        return x

    def iterate(self):
        # mutate the best-so-far individual
        mutation = self.rng_optimization.standard_normal(size=(self.ndim_problem,))
        return self.best_so_far_x + self.global_std * mutation
