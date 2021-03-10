import numpy as np


class Optimizer(object):
    """Base class of all optimizers for continuous black-box minimization.
    """
    def __init__(self, problem, options):
        # problem-related settings
        self.fitness_function = problem.get("fitness_function")
        self.ndim_problem = problem["ndim_problem"]
        self.upper_boundary = problem.get("upper_boundary")
        self.lower_boundary = problem.get("lower_boundary")
        self.initial_upper_boundary = problem.get("initial_upper_boundary", self.upper_boundary)
        self.initial_lower_boundary = problem.get("initial_lower_boundary", self.lower_boundary)
        self.problem_name = problem.get("problem_name")
        if (self.problem_name is None) and hasattr(self.fitness_function, "__name__"):
            self.problem_name = self.fitness_function.__name__

        # optimizer-related options
        self.max_function_evaluations = options.get("max_function_evaluations", np.Inf)
        self.max_runtime = options.get("max_runtime", np.Inf)
        self.fitness_threshold = options.get("fitness_threshold", -np.Inf)
        self.n_individuals = options.get("n_individuals")
        self.seed_rng = options.get("seed_rng")
        if self.seed_rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(self.seed_rng)
        self.seed_initialization = options.get("seed_initialization", self.rng.integers(np.iinfo(np.int64).max))
        self.seed_optimization = options.get("seed_optimization", self.rng.integers(np.iinfo(np.int64).max))
        self.record_options = options.get("record_options")

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def optimize(self, fitness_function=None):
        raise NotImplementedError
