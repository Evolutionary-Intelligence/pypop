import numpy as np

from pypop7.optimizers.pso.pso import PSO


class CLPSO(PSO):
    """Comprehensive Learning Particle Swarm Optimizer (CLPSO).

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
                * 'c'             - comprehensive learning rate (`float`, default: `1.49445`),
                * 'm'             - refreshing gap (`int`, default: `7`),
                * 'max_ratio_v'   - maximal ratio of velocities w.r.t. search range (`float`, default: `0.2`).

    Examples
    --------
    Use the optimizer `CLPSO` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.pso.clpso import CLPSO
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> clpso = CLPSO(problem, options)  # initialize the optimizer class
       >>> results = clpso.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CLPSO: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CLPSO: 5000, 7.184727085112434e-05

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/f3pp4nfh>`_ for more details.

    Attributes
    ----------
    c             : `float`
                    comprehensive learning rate.
    m             : `int`
                    refreshing gap.
    max_ratio_v   : `float`
                    maximal ratio of velocities w.r.t. search range.
    n_individuals : `int`
                    swarm (population) size, aka number of particles.

    References
    ----------
    Liang, J.J., Qin, A.K., Suganthan, P.N. and Baskar, S., 2006.
    Comprehensive learning particle swarm optimizer for global optimization of multimodal functions.
    IEEE Transactions on Evolutionary Computation, 10(3), pp.281-295.
    https://ieeexplore.ieee.org/abstract/document/1637688

    See the original MATLAB source code from Suganthan:
    https://github.com/P-N-Suganthan/CODES/blob/master/2006-IEEE-TEC-CLPSO.zip
    """
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)
        self.c = options.get('c', 1.49445)  # comprehensive learning rate
        self.m = options.get('m', 7)  # refreshing gap
        pc = 5.0*np.linspace(0, 1, self.n_individuals)
        self._pc = 0.5*(np.exp(pc) - np.exp(pc[0]))/(np.exp(pc[-1]) - np.exp(pc[0]))
        # set number of successive generations each particle has not improved its best fitness
        self._flag = np.zeros((self.n_individuals,))
        # set linearly decreasing inertia weights from 0.9 to 0.2
        self._w = 0.9 - 0.7*(np.arange(self._max_generations) + 1.0)/self._max_generations

    def _learn_topology(self, p_x, p_y, i, n_x):
        if self._flag[i] >= self.m:  # if no improved for successive generations
            self._flag[i], exemplars = 0, i*np.ones((self.ndim_problem,))
            for d in range(self.ndim_problem):
                if self.rng_optimization.random() < self._pc[i]:  # learn from other
                    # use tournament selection
                    left, right = self.rng_optimization.choice(self.n_individuals, 2, replace=False)
                    if p_y[left] < p_y[right]:
                        n_x[i, d], exemplars[d] = p_x[left, d], left
                    else:
                        n_x[i, d], exemplars[d] = p_x[right, d], right
                else:  # inherit from its own best position
                    n_x[i, d] = p_x[i, d]
            if np.alltrue(exemplars == i):  # learning from other when all exemplars are itself
                ndim = self.rng_optimization.integers(self.ndim_problem)  # randomly selected dimension
                exemplar = self.rng_optimization.choice(np.setdiff1d(range(self.n_individuals), i))
                n_x[i, ndim] = p_x[exemplar, ndim]

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None):
        for i in range(self.n_individuals):  # online (rather batch) update
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x
            self._learn_topology(p_x, p_y, i, n_x)
            v[i] = (self._w[self._n_generations]*v[i] + self.c*self.rng_optimization.uniform(
                size=(self.ndim_problem,))*(n_x[i] - x[i]))  # velocity update
            min_v, max_v = v[i] < self._min_v, v[i] > self._max_v  # velocity limitations
            v[i, min_v], v[i, max_v] = self._min_v[min_v], self._max_v[max_v]
            x[i] += v[i]  # position update
            y[i] = self._evaluate_fitness(x[i], args)
            if y[i] < p_y[i]:  # personally-best position update
                p_x[i], p_y[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary), y[i]
            else:
                self._flag[i] += 1
        self._n_generations += 1
        return v, x, y, p_x, p_y, n_x
