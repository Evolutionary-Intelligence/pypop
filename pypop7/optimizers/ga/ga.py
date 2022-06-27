import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class GA(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self._n_generations = 0
        self._n_restarts = 0
        self.x = options.get('x')
        self.prob_mutate = options.get('prob_mutate', 0.5)
        self.prob_cross = options.get('prob_cross', 0.5)
        if self.n_individuals is None:  # number of offspring, offspring population size (λ: lambda)
            self.n_individuals = 4 + int(3 * np.log(self.ndim_problem))  # for small population setting
        if self.n_parents is None:  # number of parents, parental population size (μ: mu)
            self.n_parents = int(self.n_individuals / 2)

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def crossover(self, x, y, cross_type):
        s1 = x.copy()
        s2 = y.copy()
        if cross_type == 'one_point':
            place = np.random.randint(0, self.ndim_problem)
            for i in range(place):
                s1[i] = y[i]
                s2[i] = x[i]
        elif cross_type == 'two_point':
            place1 = np.random.randint(0, self.ndim_problem)
            place2 = np.random.randint(place1, self.ndim_problem)
            for i in range(place1, place2):
                s1[i] = y[i]
                s2[i] = x[i]
        elif cross_type == 'uniform':
            for i in range(self.ndim_problem):
                rand = np.random.random()
                if rand < 0.5:
                    s1[i] = y[i]
                    s2[i] = x[i]
        return s1, s2

    def mutate(self, x):
        for i in range(self.ndim_problem):
            rand = np.random.random()
            if rand < self.prob_mutate:
                x[i] = self.lower_boundary[i] + np.random.random() *\
                       (self.upper_boundary[i] - self.lower_boundary[i])
        return x

    def _initialize_x(self, is_restart=False):
        if is_restart or (self.x is None):
            x = self.rng_initialization.uniform(self.initial_lower_boundary,
                                                   self.initial_upper_boundary)
        else:
            x = np.copy(self.x)
        return x

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose_frequency):
            best_so_far_y = -self.best_so_far_y if self._is_maximization else self.best_so_far_y
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect_results(self, fitness, mean=None):
        results = Optimizer._collect_results(self, fitness)
        results['_n_generations'] = self._n_generations
        return results
