import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer
from pypop7.optimizers.pso.spso import PSO


class CPSO(PSO):
    """Cooperative Particle Swarm Optimizer (CPSO).

    Parameters
    ----------
    problem : dict
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : dict
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - swarm (population) size, aka number of particles (`int`, default: `20`),
                * 'constriction'  - constriction factor (`float`, default: `0.729`),
                * 'cognition'     - cognitive learning rate (`float`, default: `2.05`),
                * 'society'       - social learning rate (`float`, default: `2.05`),
                * 'max_ratio_v'   - maximal ratio of velocities w.r.t. search range (`float`, default: `0.5`).


    Attributes
    ----------
    cognition     : `float`
                    cognitive learning rate, aka acceleration coefficient.
    max_ratio_v   : `float`
                    maximal ratio of velocities w.r.t. search range.
    n_individuals : `int`
                    swarm (population) size, aka number of particles.
    society       : `float`
                    social learning rate, aka acceleration coefficient.

    References
    ----------
    Van den Bergh, F. and Engelbrecht, A.P., 2004.
    A cooperative approach to particle swarm optimization.
    IEEE transactions on evolutionary computation, 8(3), pp.225-239.
    https://ieeexplore.ieee.org/document/1304845
    """
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)
        self.cognition = options.get('cognition', 1.49)  # cognitive learning rate
        self.society = options.get('society', 1.49)  # social learning rate
        self._max_generations = np.ceil(self.max_function_evaluations / self.n_individuals)
        self._w = 1 - (np.arange(self._max_generations) + 1.0) / self._max_generations  # from 1 to 0

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None):
        for j in range(self.ndim_problem):
            cognition_rand = self.rng_optimization.uniform(size=(self.n_individuals, self.ndim_problem))
            society_rand = self.rng_optimization.uniform(size=(self.n_individuals, self.ndim_problem))
            for i in range(self.n_individuals):
                if self._check_terminations():
                    return v, x, y, p_x, p_y, n_x

                n_x[i, j] = p_x[np.argmin(p_y), j]
                v[i, j] = (self._w[self._n_generations] * v[i, j] +
                        self.cognition * cognition_rand[i, j] * (p_x[i, j] - x[i, j]) +
                        self.society * society_rand[i, j] * (n_x[i, j] - x[i, j]))  # velocity update
                v[i, j] = np.clip(v[i, j], self._min_v[j], self._max_v[j])
                x[i, j] += v[i, j]  # position update
                y[i] = self._evaluate_fitness(x[i], args)  # fitness evaluation
                if y[i] < p_y[i]:  # online update
                    p_x[i, j], p_y[i] = x[i, j], y[i]

            self._n_generations += 1
        return v, x, y, p_x, p_y, n_x

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        v, x, y, p_x, p_y, n_x = self.initialize(args)
        while not self.termination_signal:
            self._print_verbose_info(fitness, y)
            v, x, y, p_x, p_y, n_x = self.iterate(v, x, y, p_x, p_y, n_x, args)
        return self._collect_results(fitness, y)
