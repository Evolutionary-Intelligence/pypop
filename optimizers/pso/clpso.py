import numpy as np

from optimizers.pso.pso import PSO


class CLPSO(PSO):
    """Comprehensive Learning Particle Swarm Optimizer (CLPSO).

    Reference
    ---------
    Liang, J.J., Qin, A.K., Suganthan, P.N. and Baskar, S., 2006.
    Comprehensive learning particle swarm optimizer for global optimization of multimodal functions.
    IEEE Transactions on Evolutionary Computation, 10(3), pp.281-295.
    https://ieeexplore.ieee.org/abstract/document/1637688

    Original MATLAB Source Code:
    https://www3.ntu.edu.sg/home/epnsugan/
    https://github.com/P-N-Suganthan/CODES/blob/master/2006-IEEE-TEC-CLPSO.zip
    """
    def __init__(self, problems, options):
        PSO.__init__(self, problems, options)
        # comprehensive-learning rate including cognition-learning and society-learning
        self.c = options.get('c', 1.49445)
        # refreshing gap (m)
        # note that it is same with the original code (5) but different from the original paper (7)
        self.m = options.get('m', 5)
        # learning rates for all particles (pc -> Pc)
        # note that it is same with the original code but different from the original paper
        pc = 5 * np.hstack((np.arange(0, 1, 1 / (self.n_individuals - 1)), 1))
        self.Pc = 0.5 * (np.exp(pc) - np.exp(pc[0])) / (np.exp(pc[-1]) - np.exp(pc[0]))
        # number of generations each particle has not improved its own personally previous-best fitness
        self.flag = np.zeros((self.n_individuals,))
        # linearly decreasing inertia weight
        # note that it is same with the original code but different from the original paper
        self.w = 0.9 - np.arange(1, self.max_generations + 1) * (0.7 / self.max_generations)
        self.topology = self.learning_topology

    def learning_topology(self, p_x, p_y, i, n_x):
        # note that for clearer coding it is slightly different from the original code
        # which used the vector-based form (hard to read), but it is same with the original paper
        if self.flag[i] >= self.m:
            self.flag[i] = 0
            all_exemplars = i * np.ones((self.ndim_problem,))
            for d in range(self.ndim_problem):
                if self.rng_optimization.random() < self.Pc[i]:
                    # tournament selection
                    left, right = self.rng_optimization.choice(self.n_individuals, 2, replace=False)
                    if p_y[left] < p_y[right]:
                        n_x[i, d] = p_x[left, d]
                        all_exemplars[d] = left
                    else:
                        n_x[i, d] = p_x[right, d]
                        all_exemplars[d] = right
                else:
                    n_x[i, d] = p_x[i, d]
            if np.alltrue(all_exemplars == i):
                possible_exemplars = set(range(self.n_individuals))
                possible_exemplars.remove(i)
                exemplar_dim = self.rng_optimization.integers(self.ndim_problem)
                exemplar = self.rng_optimization.choice(list(possible_exemplars))
                n_x[i, exemplar_dim] = p_x[exemplar, exemplar_dim]
        return n_x[i], i

    def initialize(self):
        x, y, p_x, p_y, n_x, v = PSO.initialize(self)
        # evaluate fitness in advance
        self.record_fitness_initialization = True
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y, p_x, p_y, n_x, v
            y[i] = self._evaluate_fitness(x[i])
        p_y = np.copy(y)
        self.n_generations += 1
        return x, y, p_x, p_y, n_x, v

    def iterate(self, x=None, y=None, p_x=None, p_y=None, n_x=None, v=None):
        # use online (rather batch) update
        for i in range(self.n_individuals):
            # update neighbor topology
            n_x[i], _ = self.topology(p_x, p_y, i, n_x)
            # update and limit velocities of particles
            learning_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            v[i] = self.w[self.n_generations] * v[i] + self.c * learning_rand * (n_x[i] - x[i])
            v_i = v[i]
            v_i[v_i > self.max_v] = self.max_v[v_i > self.max_v]
            v_i[v_i < self.min_v] = self.min_v[v_i < self.min_v]
            # update positions of particles
            # note that the original code does not limit positions in feasible search space
            x[i] += v[i]
            # evaluate fitness
            if self._check_terminations():
                return x, y, p_x, p_y, n_x, v
            y[i] = self._evaluate_fitness(x[i])
            if y[i] < p_y[i]:
                p_x[i], p_y[i] = x[i], y[i]
                # note that it is same with the original code but different from the original paper
                # self.flag[i] = 0
            else:
                self.flag[i] += 1
        return x, y, p_x, p_y, n_x, v
