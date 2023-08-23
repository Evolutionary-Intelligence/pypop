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
                * 'cognition'     - cognitive learning rate (`float`, default: `1.49`),
                * 'society'       - social learning rate (`float`, default: `1.49`),
                * 'max_ratio_v'   - maximal ratio of velocities w.r.t. search range (`float`, default: `0.2`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.pso.cpso import CPSO
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> cpso = CPSO(problem, options)  # initialize the optimizer class
       >>> results = cpso.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CPSO: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CPSO: 5000, 0.3085868239334274

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/27nx42rm>`_ for more details.

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
    IEEE Transactions on Evolutionary Computation, 8(3), pp.225-239.
    https://ieeexplore.ieee.org/document/1304845
    """
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)
        self.cognition = options.get('cognition', 1.49)  # cognitive learning rate
        assert self.cognition >= 0.0
        self.society = options.get('society', 1.49)  # social learning rate
        assert self.society >= 0.0
        self._max_generations = np.ceil(self.max_function_evaluations/(self.n_individuals*self.ndim_problem))
        self._w = 1.0 - (np.arange(self._max_generations) + 1.0)/self._max_generations  # from 1.0 to 0.0

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None):
        fitness = []
        for j in range(self.ndim_problem):
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x, fitness
            cognition_rand = self.rng_optimization.uniform(size=(self.n_individuals, self.ndim_problem))
            society_rand = self.rng_optimization.uniform(size=(self.n_individuals, self.ndim_problem))
            for i in range(self.n_individuals):
                if self._check_terminations():
                    return v, x, y, p_x, p_y, n_x, fitness
                n_x[i, j] = p_x[np.argmin(p_y), j]
                v[i, j] = (self._w[min(self._n_generations, len(self._w))]*v[i, j] +
                           self.cognition*cognition_rand[i, j]*(p_x[i, j] - x[i, j]) +
                           self.society*society_rand[i, j]*(n_x[i, j] - x[i, j]))  # velocity update
                v[i, j] = np.clip(v[i, j], self._min_v[j], self._max_v[j])
                x[i, j] += v[i, j]  # position update
                xx = np.copy(self.best_so_far_x)
                xx[j] = x[i, j]
                y[i] = self._evaluate_fitness(xx, args)
                fitness.append(y[i])
                if y[i] < p_y[i]:  # online update
                    p_x[i, j], p_y[i] = x[i, j], y[i]
        self._n_generations += 1
        return v, x, y, p_x, p_y, n_x, fitness

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        v, x, y, p_x, p_y, n_x = self.initialize(args)
        yy = y  # only for printing
        while not self.termination_signal:
            self._print_verbose_info(fitness, yy)
            v, x, y, p_x, p_y, n_x, yy = self.iterate(v, x, y, p_x, p_y, n_x, args)
        return self._collect(fitness, yy)
