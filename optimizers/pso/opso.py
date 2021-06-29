import numpy as np

from optimizers.pso.pso import PSO


class OPSO(PSO):
    """Online Particle Swarm Optimizer (OPSO).

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

    MATLAB Source Code:
    https://github.com/P-N-Suganthan/CODES/blob/master/2006-IEEE-TEC-CLPSO.zip
    """
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)

    def initialize(self, args=None):
        x, y, p_x, p_y, n_x, v = PSO.initialize(self)
        self.record_fitness_initialization = True  # evaluate fitness in advance
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y, p_x, p_y, n_x, v
            y[i] = self._evaluate_fitness(x[i], args)
        p_y = np.copy(y)
        self.n_generations += 1
        return x, y, p_x, p_y, n_x, v

    def iterate(self, x=None, y=None, p_x=None, p_y=None, n_x=None, v=None, args=None):  # use online update
        rng = self.rng_optimization
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y, p_x, p_y, n_x, v
            # evaluate fitness
            y[i] = self._evaluate_fitness(x[i], args)
            if y[i] < p_y[i]:
                p_x[i], p_y[i] = x[i], y[i]
            # update neighbor topology
            n_x[i], _ = self.topology(p_x, p_y, i)
            # update and limit velocity
            cognition_rand = rng.uniform(size=(self.ndim_problem,))
            society_rand = rng.uniform(size=(self.ndim_problem,))
            v[i] = self.w[self.n_generations] * v[i] +\
                self.cognition * cognition_rand * (p_x[i] - x[i]) +\
                self.society * society_rand * (n_x[i] - x[i])
            less_min_v, more_max_v = v[i] < self.min_v, v[i] > self.max_v
            v[i, less_min_v], v[i, more_max_v] = self.min_v[less_min_v], self.max_v[more_max_v]
            # update position
            x[i] += v[i]
        return x, y, p_x, p_y, n_x, v
