import numpy as np

from optimizers.rs.rhc import RHC


class ARHC(RHC):
    """Annealed Random (Stochastic) Hill Climber (ARHC).

    Reference
    ---------
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/hillclimber.py
    """
    def __init__(self, problem, options):
        RHC.__init__(self, problem, options)
        self.temperature = options.get('temperature', 1.0)
        self.parent_x = np.copy(self.best_so_far_x)
        self.parent_y = np.copy(self.best_so_far_y)

    def iterate(self):
        mutation = self.rng_optimization.standard_normal(size=(self.ndim_problem,))
        return self.parent_x + self.global_std * mutation

    def _evaluate_fitness(self, x):
        y = RHC._evaluate_fitness(self, x)
        # update parent solution and fitness
        if y < self.parent_y:
            self.parent_y = y
            self.parent_x = np.copy(x)
        else:
            accept_prob = np.exp(-np.abs(y - self.parent_y) / self.temperature)
            if self.rng_optimization.random() < accept_prob:
                self.parent_y = y
                self.parent_x = np.copy(x)
        return y
