import numpy as np

from pypop7.optimizers.pso.pso import PSO


class IPSO(PSO):
    """Incremental particle swarm optimizer (IPSO).

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
                * 'max_runtime'              - maximal runtime (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - swarm (population) size, number of particles (`int`, default: `20`),
                * 'cognition'     - cognitive learning rate (`float`, default: `2.0`),
                * 'society'       - social learning rate (`float`, default: `2.0`),
                * 'max_ratio_v'   - maximal ratio of velocities w.r.t. search range (`float`, default: `0.2`).

    Attributes
    ----------
    n_individuals : `int`
                    swarm (population) size, number of particles.
    cognition     : `float`
                    cognitive learning rate.
    society       : `float`
                    social learning rate.
    max_ratio_v   : `float`
                    maximal ratio of velocities w.r.t. search range.

    References
    ----------
    De Oca, M.A.M., Stutzle, T., Van den Enden, K. and Dorigo, M., 2010.
    Incremental social learning in particle swarms.
    IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 41(2), pp.368-384.
    https://ieeexplore.ieee.org/document/5582312
    """

    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)
        self.n_individuals = 1
        self.min_n_individuals = 1
        self.max_n_individuals = 1000
        self.constriction_factor = 0.729

    def initialize(self, args=None):
        v = np.zeros((self.n_individuals, self.ndim_problem))  # velocities
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=self._swarm_shape)  # positions
        y = np.empty((self.n_individuals,))  # fitness
        p_x, p_y = np.copy(x), np.copy(y)  # personally previous-best positions and fitness
        n_x = np.copy(x)  # neighborly previous-best positions
        for i in range(self.n_individuals):
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x
            y[i] = self._evaluate_fitness(x[i], args)
        p_y = np.copy(y)
        return v, x, y, p_x, p_y, n_x

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None):
        # Horizontal social learning
        for i in range(self.n_individuals):
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x
            n_x[i] = p_x[np.argmin(p_y)]  # online update within global topology
            cognition_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            society_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            v[i] = self.constriction_factor * (v[i] +
                                               self.cognition * cognition_rand * (p_x[i] - x[i]) +
                                               self.society * society_rand * (n_x[i] - x[i]))  # velocity update
            min_v, max_v = v[i] < self._min_v, v[i] > self._max_v
            v[i, min_v], v[i, max_v] = self._min_v[min_v], self._max_v[max_v]
            x[i] += v[i]  # position update
            y[i] = self._evaluate_fitness(x[i], args)  # fitness evaluation
            if y[i] < p_y[i]:  # online update
                p_x[i], p_y[i] = x[i], y[i]

        # Population growth and vertical social learning
        if self.n_individuals < self.max_n_individuals:
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x
            xx = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
            xx = xx + self.rng_initialization.uniform() * (n_x[-1] - xx)
            yy = self._evaluate_fitness(xx, args)  # fitness evaluation
            v = np.vstack((v, np.zeros((self.ndim_problem,))))
            x = np.vstack((x, xx))
            y = np.hstack((y, yy))
            p_x = np.vstack((p_x, xx))
            p_y = np.hstack((p_y, yy))
            n_x = np.vstack((n_x, p_x[np.argmin(p_y)]))
            self.n_individuals += 1
        self._n_generations += 1
        return v, x, y, p_x, p_y, n_x
