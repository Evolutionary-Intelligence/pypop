import time
import numpy as np

from optimizers.core.optimizer import Optimizer


class PSO(Optimizer):
    """Particle Swarm Optimizer (PSO).

    Reference
    ---------
    Kennedy, J. and Eberhart, R., 1995, November.
    Particle swarm optimization.
    In Proceedings of International Conference on Neural Networks (Vol. 4, pp. 1942-1948). IEEE.
    https://ieeexplore.ieee.org/document/488968

    Shi, Y. and Eberhart, R., 1998, May.
    A modified particle swarm optimizer.
    In IEEE World Congress on Computational Intelligence (pp. 69-73). IEEE.
    https://ieeexplore.ieee.org/abstract/document/699146

    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/populationbased/pso.py
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # swarm (population) size
            self.n_individuals = 20  # number of particles
        self.swarm_size = (self.n_individuals, self.ndim_problem)
        self.cognition = options.get('cognition', 2.0)  # cognition-learning rate
        self.society = options.get('society', 2.0)  # society-learning rate
        self.topology = None  # to control neighbors of society learning
        self.max_ratio_v = options.get('max_ratio_v', 0.2)  # maximal ratio of velocities w.r.t. entire search range
        self.max_v = self.max_ratio_v * (self.upper_boundary - self.lower_boundary)
        self.min_v = -self.max_v
        self.n_generations = options.get('n_generations', 0)
        self.max_generations = int(np.ceil(self.max_function_evaluations / self.n_individuals))
        w = 0.9 - np.arange(1, self.max_generations + 1) * 0.5 / self.max_generations
        self.w = options.get('w', w)  # inertia weight
        self.record_fitness_initialization = False  # record all fitness generated during initialization

    def initialize(self, args=None):
        rng = self.rng_initialization
        x = rng.uniform(self.initial_lower_boundary, self.initial_upper_boundary, size=self.swarm_size)  # positions
        y = np.empty((self.n_individuals,))  # fitness
        p_x, p_y = np.copy(x), np.copy(y)  # personally previous-best positions and fitness
        n_x = np.copy(x)  # neighborly previous-best positions
        v = rng.uniform(self.min_v, self.max_v, size=self.swarm_size)  # velocities
        return x, y, p_x, p_y, n_x, v

    def iterate(self, x=None, y=None, p_x=None, p_y=None, n_x=None, v=None, args=None):  # use batch update
        rng = self.rng_optimization
        # evaluate fitness
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y, p_x, p_y, n_x, v
            y[i] = self._evaluate_fitness(x[i], args)
            if y[i] < p_y[i]:
                p_x[i], p_y[i] = np.copy(x[i]), y[i]
        # update neighbor topology of each particle
        for i in range(self.n_individuals):
            n_x[i], _ = self.topology(p_x, p_y, i)
        # update and limit velocities of particles
        cognition_rand, society_rand = rng.uniform(size=self.swarm_size), rng.uniform(size=self.swarm_size)
        v = self.w[self.n_generations] * v +\
            self.cognition * cognition_rand * (p_x - x) +\
            self.society * society_rand * (n_x - x)
        for i in range(self.n_individuals):
            less_min_v, more_max_v = v[i] < self.min_v, v[i] > self.max_v
            v[i, less_min_v] = self.min_v[less_min_v]
            v[i, more_max_v] = self.max_v[more_max_v]
        # update positions of particles
        x += v
        return x, y, p_x, p_y, n_x, v

    def optimize(self, fitness_function=None, args=None):
        self.start_time = time.time()
        fitness = []  # store all fitness generated during evolution
        if fitness_function is not None:
            self.fitness_function = fitness_function
        x, y, p_x, p_y, n_x, v = self.initialize(args)
        if self.record_fitness_initialization and self.record_options['record_fitness']:
            fitness.extend(y)
        while True:
            x, y, p_x, p_y, n_x, v = self.iterate(x, y, p_x, p_y, n_x, v, args)
            if self.record_options['record_fitness']:
                fitness.extend(y)
            if self._check_terminations():
                break
            self.n_generations += 1
            self._print_verbose_info(y)
        if self.record_options['record_fitness']:
            self._compress_fitness(fitness[:self.n_function_evaluations])
        return self._collect_results()

    def _print_verbose_info(self, y=None):
        if self.verbose_options['verbose']:
            if not self.n_generations % self.verbose_options['frequency_verbose']:
                info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
                print(info.format(self.n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))
