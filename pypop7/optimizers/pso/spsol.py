import numpy as np

from pypop7.optimizers.pso.pso import PSO


class SPSOL(PSO):
    """Standard Particle Swarm Optimizer with a Local (ring) topology (SPSOL).

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

                * 'verbose'                  - flag to print verbose information during optimization (`bool`, default:
                  `True`),
                * 'verbose_frequency'        - generation frequency of printing verbose information (`int`, default:
                  `10`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - swarm (population) size, number of particles (`int`, default: `20`),
                * 'cognition'     - cognitive learning rate (`float`, default: `2.0`),
                * 'society'       - social learning rate (`float`, default: `2.0`),
                * 'max_ratio_v'   - maximal ratio of velocities w.r.t. search range (`float`, default: `0.2`).

    Examples
    --------
    Use the PSO optimizer `SPSOL` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.pso.spsol import SPSOL
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> spsol = SPSOL(problem, options)  # initialize the optimizer class
       >>> results = spsol.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"SPSOL: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       SPSOL: 5000, 3.470837498146212e-08

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
    Shi, Y. and Eberhart, R., 1998, May.
    A modified particle swarm optimizer.
    In IEEE World Congress on Computational Intelligence (pp. 69-73). IEEE.
    https://ieeexplore.ieee.org/abstract/document/699146

    Kennedy, J. and Eberhart, R., 1995, November.
    Particle swarm optimization.
    In Proceedings of International Conference on Neural Networks (Vol. 4, pp. 1942-1948). IEEE.
    https://ieeexplore.ieee.org/document/488968
    """
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)
        assert self.n_individuals >= 3  # for ring topology

    def _ring_topology(self, p_x=None, p_y=None, i=None):
        left, right = i - 1, i + 1
        if i == 0:
            left = self.n_individuals - 1
        elif i == self.n_individuals - 1:
            right = 0
        ring = [left, i, right]
        return p_x[ring[int(np.argmin(p_y[ring]))]]

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x
            n_x[i] = self._ring_topology(p_x, p_y, i)  # online update within ring topology
            cognition_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            society_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            v[i] = (self._w[self._n_generations]*v[i] +
                    self.cognition*cognition_rand*(p_x[i] - x[i]) +
                    self.society*society_rand*(n_x[i] - x[i]))  # velocity update
            min_v, max_v = v[i] < self._min_v, v[i] > self._max_v
            v[i, min_v], v[i, max_v] = self._min_v[min_v], self._max_v[max_v]
            x[i] += v[i]  # position update
            y[i] = self._evaluate_fitness(x[i], args)  # fitness evaluation
            if y[i] < p_y[i]:  # online update
                p_x[i], p_y[i] = x[i], y[i]
        self._n_generations += 1
        return v, x, y, p_x, p_y, n_x
