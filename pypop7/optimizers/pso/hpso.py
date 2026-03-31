import numpy as np  # engine for numerical computing
from collections import deque

from pypop7.optimizers.pso.pso import PSO  # abstract class of all particle swarm optimizer (PSO) classes


class HPSO(PSO):
    """History-based Particle Swarm Optimizer (HPSO).

    Variant of standard global-topology PSO with an extra velocity term pulling each particle toward the
    mean of the swarm global-best positions recorded over the last ``n_history`` completed generations.

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
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals'  - swarm (population) size, aka number of particles (`int`, default: `20`),
                * 'cognition'      - cognitive learning rate (`float`, default: `2.0`),
                * 'society'        - social learning rate (`float`, default: `2.0`),
                * 'max_ratio_v'    - maximal ratio of velocities w.r.t. search range (`float`, default: `0.2`),
                * 'n_history'      - number of past generations whose global-best positions are averaged (`int`,
                                     default: `10`),
                * 'history_weight' - acceleration weight for the history-mean term (`float`, default: `0.5`).

    Examples
    --------
    Use the optimizer `HPSO` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.pso.hpso import HPSO
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5.0*numpy.ones((2,)),
       ...            'upper_boundary': 5.0*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> hpso = HPSO(problem, options)  # initialize the optimizer class
       >>> results = hpso.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"HPSO: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       HPSO: 5000, 3.111603118059686e-06

    Attributes
    ----------
    cognition        : `float`
                       cognitive learning rate, aka acceleration coefficient.
    history_weight   : `float`
                       weight on the velocity term toward the mean of recent global-best positions.
    max_ratio_v      : `float`
                       maximal ratio of velocities w.r.t. search range.
    n_history        : `int`
                       length of the global-best history buffer (generations).
    n_individuals    : `int`
                       swarm (population) size, aka number of particles.
    society          : `float`
                       social learning rate, aka acceleration coefficient.

    References
    ----------
    Kennedy, J. and Eberhart, R., 1995, November.
    Particle swarm optimization.
    In Proceedings of International Conference on Neural Networks (pp. 1942-1948). IEEE.
    https://ieeexplore.ieee.org/document/488968
    """
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)
        self.n_history = options.get('n_history', 10)
        assert self.n_history >= 1
        self.history_weight = options.get('history_weight', 0.5)
        assert self.history_weight >= 0.0
        self._g_history = deque(maxlen=self.n_history)

    def initialize(self, args=None):
        v, x, y, p_x, p_y, n_x = PSO.initialize(self, args)
        self._g_history.clear()
        self._g_history.append(np.copy(p_x[np.argmin(p_y)]))
        return v, x, y, p_x, p_y, n_x

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None):
        # Mean of stored global-best positions from prior completed generations (HPSO extra term).
        # If history was cleared unexpectedly, fall back to the current global best (same as n_x target).
        if len(self._g_history) == 0:
            g_bar = p_x[np.argmin(p_y)]
        else:
            g_bar = np.mean(np.stack(tuple(self._g_history), axis=0), axis=0)
        # One synchronous generation: update every particle once (same order as SPSO).
        for i in range(self.n_individuals):
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x
            n_x[i] = p_x[np.argmin(p_y)]  # online update within global topology
            # Stochastic weights per dimension, same style as standard PSO.
            cognition_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            society_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            history_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            # Linear inertia schedule (Shi & Eberhart); index matches SPSO (incremented after the generation).
            w = self._w[min(self._n_generations, len(self._w) - 1)]
            v[i] = (w*v[i] +  # inertia
                    self.cognition*cognition_rand*(p_x[i] - x[i]) +  # cognitive (personal best)
                    self.society*society_rand*(n_x[i] - x[i]) +  # social (global best this step)
                    self.history_weight*history_rand*(g_bar - x[i]))  # pull toward mean of past global bests
            v[i] = np.clip(v[i], self._min_v, self._max_v)
            x[i] += v[i]  # position update
            if self.is_bound:
                x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)
            y[i] = self._evaluate_fitness(x[i], args)  # fitness evaluation
            if y[i] < p_y[i]:  # online update
                p_x[i], p_y[i] = x[i], y[i]
        # Record this generation's global best for future g_bar averages (deque drops oldest beyond n_history).
        self._g_history.append(np.copy(p_x[np.argmin(p_y)]))
        self._n_generations += 1
        return v, x, y, p_x, p_y, n_x
