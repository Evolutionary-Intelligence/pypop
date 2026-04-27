import numpy as np
from collections import deque

from pypop7.optimizers.pso.pso import PSO


class GHEPSO(PSO):
    """Group Historical Experience Particle Swarm Optimizer (GHEPSO).

    Implements the algorithm proposed in Yan, Li & Deng (2012). In standard
    PSO the social term pulls each particle toward the *current* generation's
    global best (gbest). GHEPSO replaces that single attractor with ``p_vd``,
    the running mean of the last ``n_history`` completed-generation gbest
    positions, so that particles are guided by accumulated group experience
    rather than just the most recent optimum.

    Velocity update (eq. 5 in the paper):

        v_i(k+1) = w * v_i(k)
                 + c1 * r1 * (p_i(k)  - x_i(k))   # cognitive: personal best
                 + c2 * r2 * (p_vd(k) - x_i(k))   # social: history mean

    where:

        p_vd(k) = [ p_g(k) + p_g(k-1) + ... + p_g(k-N+1) ] / N

    and ``p_g(k)`` is the gbest position recorded at the end of generation k.
    When N=1 the algorithm reduces to standard PSO.

    The paper reports best results with N equal to the swarm size (default 20),
    fixed acceleration coefficients c1 = c2 = 2.0, and fixed inertia weight
    w = 0.7.  The base-class linear inertia schedule (Shi & Eberhart, 1998) is
    *not* part of GHEPSO; pass ``inertia_weight=0.7`` in ``options`` to match
    the paper exactly.

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
                * 'cognition'      - cognitive learning rate c1 (`float`, default: `2.0`),
                * 'society'        - social learning rate c2 applied to history mean (`float`, default: `2.0`),
                * 'max_ratio_v'    - maximal ratio of velocities w.r.t. search range (`float`, default: `0.2`),
                * 'n_history'      - number of past generations whose global-best positions are averaged,
                                     N in the paper (`int`, default: `20`).
                * 'inertia_weight' - fixed inertia weight w (`float`, default: `None`).
                                     When None the base-class linear decay schedule is used.
                                     Original GHEPSO does not use linear inertia schedule.

    Examples
    --------
    Use the optimizer `GHEPSO` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock
       >>> from pypop7.optimizers.pso.ghepso import GHEPSO
       >>> problem = {'fitness_function': rosenbrock,
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5.0*numpy.ones((2,)),
       ...            'upper_boundary': 5.0*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,
       ...            'seed_rng': 2022,
       ...            'inertia_weight': 0.7}
       >>> ghepso = GHEPSO(problem, options)
       >>> results = ghepso.optimize()
       >>> print(f"GHEPSO: {results['n_function_evaluations']}, {results['best_so_far_y']}")

    Attributes
    ----------
    cognition     : `float`
                    Cognitive learning rate c1.
    society       : `float`
                    Social learning rate c2, applied to the history-mean target p_vd.
    max_ratio_v   : `float`
                    Maximal velocity as a fraction of the search range.
    n_history     : `int`
                    Length N of the gbest history buffer (number of generations).
    n_individuals : `int`
                    Swarm size.
    inertia_weight : `float` or None
                    Fixed inertia weight, or None to use the base-class schedule.

    References
    ----------
    Yan, Z., Li, B. and Deng, C., 2012.
    A PSO algorithm based on group history experience.
    In Proceedings of the 10th World Congress on Intelligent Control and
    Automation (pp. 4108-4112). IEEE.
    https://ieeexplore.ieee.org/document/6359163
    """


    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)
        self.n_history = options.get('n_history', 20)
        assert self.n_history >= 1
        self.inertia_weight = options.get('inertia_weight', None)
        self._g_history = deque(maxlen=self.n_history)

    def initialize(self, args=None):
        v, x, y, p_x, p_y, n_x = PSO.initialize(self, args)
        self._g_history.clear()
        self._g_history.append(np.copy(p_x[np.argmin(p_y)]))
        return v, x, y, p_x, p_y, n_x

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None):
        # mean of the last N global-best positions
        p_vd = np.mean(np.stack(tuple(self._g_history), axis=0), axis=0)

        for i in range(self.n_individuals):
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x
            # Stochastic weights per dimension, same style as standard PSO.
            cognition_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            social_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            # Linear inertia schedule (Shi & Eberhart); index matches SPSO (incremented after the generation).
            w = self._w[min(self._n_generations, len(self._w) - 1)]
            if self.inertia_weight is not None:
                w = self.inertia_weight

            v[i] = (w*v[i] +  # inertia
                    self.cognition*cognition_rand*(p_x[i] - x[i]) +  # cognitive (personal best)
                    self.society*social_rand*(p_vd - x[i]))  # pull toward mean of past global bests
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
