import random

import numpy as np

from pypop7.optimizers.pso.pso import PSO
from pypop7.optimizers.core.optimizer import Optimizer


class CCPSO2(PSO):
    """New Cooperative Coevolving Particle Swarm Optimization(CCPSO2)

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
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`),
                * 'record_fitness'           - flag to record fitness list to output results (`bool`, default: `False`),
                * 'record_fitness_frequency' - function evaluations frequency of recording (`int`, default: `1000`),

                  * if `record_fitness` is set to `False`, it will be ignored,
                  * if `record_fitness` is set to `True` and it is set to 1, all fitness generated during optimization
                    will be saved into output results.

                * 'verbose'                  - flag to print verbose info during optimization (`bool`, default: `True`),
                * 'verbose_frequency'        - frequency of printing verbose info (`int`, default: `10`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - swarm (population) size, number of particles (`int`, 20),
                * 'p'             - possibility of using cauchy to refresh swarm (`float`, default: `0.5`),

    Examples
    --------
    Use the PSO optimizer `CCPSO2` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rastrigin_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.pso.ccpso2 import CCPSO2
       >>> problem = {'fitness_function': rastrigin,  # define problem arguments
       ...            'ndim_problem': 1000,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5e6,  # set optimizer options
       ...            'fitness_threshold': 1e-10
       ...            'p': 0,
       ...            'seed_rng': 0}
       >>> ccpso2 = CCPSO2(problem, options)  # initialize the optimizer class
       >>> results = ccpso2.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CCPSO2: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CCPSO2: 2951000, 9.822542779147625e-11

    Attributes
    ----------
    n_individuals : `int`
                    swarm (population) size, number of particles.
    p             : `float`
                    possibility of using cauchy to refresh swarm

    References
    ----------
    X. D. Li, X, Yao
    Cooperatively Coevolving Particle Swarms for Large Scale Optimization
    IEEE Trans. Evol. Comput. 16(2): 210-224 (2012)
    https://ieeexplore.ieee.org/document/5910380/
    """
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)
        self.p = options.get('p', 0.5)
        self.group_size_set = [2, 5, 10, 50, 100, 200]
        self.s = random.choice(self.group_size_set)
        self.k = int(self.ndim_problem / self.s)
        self.global_improved = False
        self.dimension_indices = list(range(self.ndim_problem))
        self.x_swarm, self.x_best = None, None
        self.fx_swarm, self.fx_best = np.inf, np.inf

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=self._swarm_shape)  # positions
        y = x.copy()
        x_local = x.copy()
        self.x_swarm = x[0, :].copy()
        self.x_best = self.x_swarm.copy()
        return x, y, x_local

    def b(self, j, i, x):
        p = self.x_swarm.copy()
        for d in range(j * self.s, (j+1) * self.s):
            p[self.dimension_indices[d]] = x[i][self.dimension_indices[d]]
        return p

    def local_best(self, j, i, fy):
        v_i = fy[i][j]
        v_left, v_right = None, None
        if i != 0:
            v_left = fy[i - 1][j]
        else:
            v_left = fy[self.n_individuals - 1][j]
        if i != self.n_individuals - 1:
            v_right = fy[i + 1][j]
        else:
            v_right = fy[0][j]
        if v_i < v_left and v_i < v_right:
            return i
        elif v_i < v_right:
            if i == 0:
                return self.n_individuals - 1
            else:
                return i - 1
        else:
            if i == self.n_individuals - 1:
                return 0
            else:
                return i + 1

    def iterate(self, x=None, x_local=None, y=None, args=None):
        if self.global_improved is False:
            self.s = random.choice(self.group_size_set)
            self.k = int(self.ndim_problem / self.s)
        self.global_improved = False
        random.shuffle(self.dimension_indices)
        fx = np.inf * np.ones((self.n_individuals, self.k))
        fy = fx.copy()
        y_list = []
        for j in range(self.k):
            for i in range(self.n_individuals):
                fx[i][j] = self._evaluate_fitness(self.b(j, i, x), args)
                fy[i][j] = self._evaluate_fitness(self.b(j, i, y), args)
                y_list.extend([fx[i][j], fy[i][j]])
                if fx[i][j] < fy[i][j]:
                    for d in range(j * self.s, (j + 1) * self.s):
                        y[i][self.dimension_indices[d]] = x[i][self.dimension_indices[d]]
                        fy[i][j] = fx[i][j].copy()
                if fy[i][j] < self.fx_swarm:
                    for d in range(j * self.s, (j + 1) * self.s):
                        self.x_swarm[self.dimension_indices[d]] = y[i][self.dimension_indices[d]]
                        self.fx_swarm = fy[i][j].copy()
            for i in range(self.n_individuals):
                local = self.local_best(j, i, fy)
                for d in range(j * self.s, (j + 1) * self.s):
                    x_local[i][self.dimension_indices[d]] = y[local][self.dimension_indices[d]]
            if self.fx_swarm < self.fx_best:
                for d in range(j * self.s, (j + 1) * self.s):
                    self.x_best[self.dimension_indices[d]] = self.x_swarm[self.dimension_indices[d]]
                self.fx_best = self.fx_swarm
                self.global_improved = True
        for j in range(self.k):
            for i in range(self.n_individuals):
                for d in range(j * self.s, (j + 1) * self.s):
                    tempd = self.dimension_indices[d]
                    if np.random.random() < self.p:
                        x[i][tempd] = y[i][tempd] + self.rng_optimization.standard_cauchy()\
                                     * np.abs(y[i][tempd] - x_local[i][tempd])
                    else:
                        x[i][tempd] = x_local[i][tempd] + self.rng_optimization.standard_normal() \
                                      * np.abs(y[i][tempd] - x_local[i][tempd])
        return x, x_local, y, y_list

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        x, y, x_local = self.initialize(args)
        while True:
            x, x_local, y, y_list = self.iterate(x, x_local, y, args)
            if self.record_fitness:
                fitness.extend(y_list)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y_list)
        results = self._collect_results(fitness)
        return results
