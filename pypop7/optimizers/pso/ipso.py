import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer
from pypop7.optimizers.pso.pso import PSO


class IPSO(PSO):
    """Incremental Particle Swarm Optimizer (IPSO).

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

    Examples
    --------
    Use the optimizer `IPSO` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.pso.ipso import IPSO
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> ipso = IPSO(problem, options)  # initialize the optimizer class
       >>> results = ipso.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"IPSO: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       IPSO: 5000, 2.29225104244031e-07

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/4pk3ssrf>`_ for more details.

    Attributes
    ----------
    cognition     : `float`
                    cognitive learning rate, aka acceleration coefficient.
    constriction  : `float`
                    constriction factor.
    max_ratio_v   : `float`
                    maximal ratio of velocities w.r.t. search range.
    n_individuals : `int`
                    swarm (population) size, aka number of particles.
    society       : `float`
                    social learning rate, aka acceleration coefficient.

    References
    ----------
    De Oca, M.A.M., Stutzle, T., Van den Enden, K. and Dorigo, M., 2011.
    Incremental social learning in particle swarms.
    IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 41(2), pp.368-384.
    https://ieeexplore.ieee.org/document/5582312
    """
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)
        self.n_individuals = 1  # minimum of swarm size
        self.max_n_individuals = options.get('max_n_individuals', 1000)  # maximum of swarm size
        self.cognition = options.get('cognition', 2.05)  # cognitive learning rate
        self.society = options.get('society', 2.05)  # social learning rate
        self.constriction = options.get('constriction', 0.729)  # constriction factor
        self.max_ratio_v = options.get('max_ratio_v', 0.5)  # maximal ratio of velocity

    def initialize(self, args=None):
        v = np.zeros((self.n_individuals, self.ndim_problem))  # velocities
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=self._swarm_shape)  # positions
        y = np.empty((self.n_individuals,))  # fitness
        p_x, p_y = np.copy(x), np.copy(y)  # personally previous-best positions and fitness
        for i in range(self.n_individuals):
            if self._check_terminations():
                return v, x, y, p_x, p_y
            y[i] = self._evaluate_fitness(x[i], args)
        p_y = np.copy(y)
        return v, x, y, p_x, p_y

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, args=None, fitness=None):
        for i in range(self.n_individuals):  # horizontal social learning
            if self._check_terminations():
                return v, x, y, p_x, p_y
            cognition_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            society_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            v[i] = self.constriction*(v[i] + self.cognition*cognition_rand*(p_x[i] - x[i]) +
                                      self.society*society_rand*(p_x[np.argmin(p_y)] - x[i]))  # velocity update
            v[i] = np.clip(v[i], self._min_v, self._max_v)
            x[i] += v[i]  # position update
            x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)
            y[i] = self._evaluate_fitness(x[i], args)
            if self.saving_fitness:
                fitness.append(y[i])
            if y[i] < p_y[i]:  # online update
                p_x[i], p_y[i] = x[i], y[i]
        if self.n_individuals < self.max_n_individuals:  # population growth (vertical social learning)
            if self._check_terminations():
                return v, x, y, p_x, p_y
            xx = self.rng_optimization.uniform(self.lower_boundary, self.upper_boundary)
            model = p_x[np.argmin(p_y)]  # the best particle is used as model
            # use different random numbers of different dimensions for diversity (important),
            # which is *slightly different* from the original paper but often with better performance
            # xx += self.rng_optimization.uniform()*(model - xx)  # from the original paper
            xx += self.rng_optimization.uniform(size=(self.ndim_problem,))*(model - xx)
            xx = np.clip(xx, self.lower_boundary, self.upper_boundary)
            yy = self._evaluate_fitness(xx, args)
            if self.saving_fitness:
                fitness.append(yy)
            v = np.vstack((v, np.zeros((self.ndim_problem,))))
            x, y = np.vstack((x, xx)), np.hstack((y, yy))
            p_x, p_y = np.vstack((p_x, xx)), np.hstack((p_y, yy))
            self.n_individuals += 1
        self._n_generations += 1
        return v, x, y, p_x, p_y

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        v, x, y, p_x, p_y = self.initialize(args)
        if self.saving_fitness:
            fitness.extend(y)
        self._print_verbose_info(y)
        while True:
            v, x, y, p_x, p_y = self.iterate(v, x, y, p_x, p_y, args, fitness)
            if self._check_terminations():
                break
            self._print_verbose_info(y)
        return self._collect_results(fitness)
