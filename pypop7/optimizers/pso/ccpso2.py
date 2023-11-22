import numpy as np
from scipy.stats import cauchy

from pypop7.optimizers.core.optimizer import Optimizer
from pypop7.optimizers.pso.pso import PSO


class CCPSO2(PSO):
    """Cooperative Coevolving Particle Swarm Optimizer (CCPSO2).

    .. note:: `CCPSO2` employs the popular `cooperative coevolution <https://tinyurl.com/57wzdrhm>`_ framework to
       extend PSO for large-scale black-box optimization (LSBBO) with *random grouping/partitioning*. However, it
       may suffer from **performance degradation** on *non-separable* functions (particularly ill-conditioned),
       owing to its axis-parallel decomposition strategy (see the classical **coordinate descent** from the
       mathematical programming community for detailed mathematical explanation).

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
                * 'n_individuals' - swarm (population) size, aka number of particles (`int`, default: `30`),
                * 'p'             - probability of using Cauchy sampling distribution (`float`, default: `0.5`),
                * 'group_sizes'   - a pool of candidate dimensions for grouping (`list`, default:
                  `[2, 5, 10, 50, 100, 250]`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rastrigin_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.pso.ccpso2 import CCPSO2
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 500,
       ...            'lower_boundary': -5*numpy.ones((500,)),
       ...            'upper_boundary': 5*numpy.ones((500,))}
       >>> options = {'max_function_evaluations': 1000000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> ccpso2 = CCPSO2(problem, options)  # initialize the optimizer class
       >>> results = ccpso2.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CCPSO2: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CCPSO2: 1000000, 1150.0205163111475

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/bdfywpyx>`_ for more details.

    Attributes
    ----------
    group_sizes   : `list`
                    a pool of candidate dimensions for grouping.
    n_individuals : `int`
                    swarm (population) size, aka number of particles.
    p             : `float`
                    probability of using Cauchy sampling distribution.

    References
    ----------
    Li, X. and Yao, X., 2012.
    Cooperatively coevolving particle swarms for large scale optimization.
    IEEE Transactions on Evolutionary Computation, 16(2), pp.210-224.
    https://ieeexplore.ieee.org/document/5910380/

    Potter, M.A. and De Jong, K.A., 1994, October.
    A cooperative coevolutionary approach to function optimization.
    In International Conference on Parallel Problem Solving from Nature (pp. 249-257).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/3-540-58484-6_269
    """
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)
        self.n_individuals = options.get('n_individuals', 30)  # swarm (population) size, aka number of particles
        assert self.n_individuals > 0
        self.p = options.get('p', 0.5)  # probability of using Cauchy sampling distribution
        assert 0.0 <= self.p <= 1.0
        self.group_sizes = options.get('group_sizes', [2, 5, 10, 50, 100, 250])
        assert np.all(np.array(self.group_sizes) <= self.ndim_problem)
        self._indices = np.arange(self.ndim_problem)  # indices of all dimensions
        self._s = self.rng_optimization.choice(self.group_sizes)  # dimension to be optimized by each swarm
        self._k = int(np.ceil(self.ndim_problem/self._s))  # number of swarms
        assert self._k > 0
        self._improved = True
        self._best_so_far_y = self.best_so_far_y

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # positions
        y = np.empty((self._k, self.n_individuals))  # fitness for all individuals of all swarms
        p_x, p_y = np.copy(x), np.copy(y)  # personally best positions and fitness
        n_x = np.copy(x)  # neighborly best positions
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y, p_x, p_y, n_x
            y[:, i] = self._evaluate_fitness(x[i], args)
        p_y = np.copy(y)
        return x, y, p_x, p_y, n_x

    def _ring_topology(self, p_x=None, p_y=None, j=None, i=None, indices=None):
        left, right = i - 1, i + 1
        if i == 0:
            left = self.n_individuals - 1
        elif i == self.n_individuals - 1:
            right = 0
        ring = [left, i, right]
        return p_x[ring[int(np.argmin(p_y[j, ring]))], indices]

    def iterate(self, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None, fitness=None):
        if self._improved is False:
            candidates = self.rng_optimization.choice(self.group_sizes, size=(2,), replace=False)
            if self._s == candidates[0]:  # to ensure that a different value is chosen
                self._s = candidates[1]
            else:
                self._s = candidates[0]
            self._k = int(np.ceil(self.ndim_problem/self._s))
            self.rng_optimization.shuffle(self._indices)  # random permutation
            y = np.empty((self._k, self.n_individuals))  # fitness for all individuals of all swarms
            for j in range(self._k):  # for each swarm
                for i in range(self.n_individuals):  # for each individual
                    if self._check_terminations():
                        return x, y, p_x, p_y, n_x
                    cv = np.copy(self.best_so_far_x)  # context vector
                    indices = self._indices[np.arange(j*self._s, (j + 1)*self._s)]
                    cv[indices] = x[i, indices]
                    y[j, i] = self._evaluate_fitness(cv, args)
            if self.saving_fitness:
                fitness.extend(y.flatten())
            p_y = np.copy(y)
        self._improved, self._best_so_far_y = False, self.best_so_far_y
        for j in range(self._k):  # for each swarm
            for i in range(self.n_individuals):  # for each individual
                if self._check_terminations():
                    return x, y, p_x, p_y, n_x
                cv = np.copy(self.best_so_far_x)  # context vector
                indices = self._indices[np.arange(j*self._s, (j + 1)*self._s)]
                cv[indices] = x[i, indices]
                y[j, i] = self._evaluate_fitness(cv, args)
                if y[j, i] < p_y[j, i]:
                    p_x[i, indices], p_y[j, i] = cv[indices], y[j, i]
                if y[j, i] < self._best_so_far_y:
                    self._improved, self._best_so_far_y = True, y[j, i]
            for i in range(self.n_individuals):
                indices = self._indices[np.arange(j*self._s, (j + 1)*self._s)]
                n_x[i, indices] = self._ring_topology(p_x, p_y, j, i, indices)
        for j in range(self._k):  # for each swarm
            for i in range(self.n_individuals):  # for each individual
                indices = self._indices[np.arange(j*self._s, (j + 1)*self._s)]
                std = np.abs(p_x[i, indices] - n_x[i, indices])
                for jj, ii in enumerate(indices):
                    if self.rng_optimization.random() <= self.p:  # sampling from Cauchy distribution
                        x[i, ii] = p_x[i, ii] + std[jj]*cauchy.rvs(random_state=self.rng_optimization)
                    else:  # sampling from Gaussian distribution
                        x[i, ii] = n_x[i, ii] + std[jj]*self.rng_optimization.standard_normal()
                x[i, indices] = np.clip(x[i, indices], self.lower_boundary[indices], self.upper_boundary[indices])
        self._n_generations += 1
        return x, y, p_x, p_y, n_x

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        x, y, p_x, p_y, n_x = self.initialize(args)
        self._print_verbose_info(fitness, y[0])
        while not self.termination_signal:
            x, y, p_x, p_y, n_x = self.iterate(x, y, p_x, p_y, n_x, args, fitness)
            self._print_verbose_info(fitness, y.flatten())
        return self._collect(fitness)
