from optimizers.rs.prs import PRS


class RHC(PRS):
    """Random Hill Climber (RHC).

    Only support normally distributed random sampling during optimization.
    But support uniformly or normally distributed random sampling for the starting search point.

    Reference
    ---------
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/hillclimber.py
    """
    def __init__(self, problem, options):
        PRS.__init__(self, problem, options)
        self.initialization_distribution = options.get('initialization_distribution', 1)  # 1 -> uniformly distributed
        if self.initialization_distribution not in [0, 1]:  # 0 -> normally distributed
            info = 'Currently for optimizer {:s}, only support uniformly or normally distributed random initialization.'
            raise ValueError(info.format(self.__class__.__name__))
        if self.initialization_distribution == 0:  # only for normally distributed random initialization
            self.initial_std = options.get('initial_std', 1.0)
        self.global_std = options.get('global_std', 0.1)

    def _sample(self, rng):
        if self.initialization_distribution == 0:
            x = rng.standard_normal(size=(self.ndim_problem,)) * self.initial_std
        else:
            x = rng.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        return x

    def iterate(self):  # draw sample via mutating the best-so-far individual
        noise = self.rng_optimization.standard_normal(size=(self.ndim_problem,))
        return self.best_so_far_x + self.global_std * noise
