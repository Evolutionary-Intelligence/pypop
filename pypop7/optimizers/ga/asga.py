import numpy as np

from pypop7.optimizers.ga.ga import GA
from athena.active import ActiveSubspaces


class ASGA(GA):
    """Active Subspace Genetic Algorithm (ASGA).

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
              and with the following particular setting (`key`):
                * 'n_initial_individuals' - initial population size (`int`, default: `2000`),
                * 'n_individuals'         - population size (`int`, default: `200`),
                * 'n_subspace'            - dimensionality number of active subspaces (`int`, default: `1`),
                * 'crossover_prob'        - crossover probability (`float`, default: `0.5`),
                * 'mutation_prob'         - mutation probability (`float`, default: `0.5`),
                * 'b'                     - number of back-mapped points (`int`, default: `2`).

    Examples
    --------
    Use the optimizer `ASGA` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.ga.asga import ASGA
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> asga = ASGA(problem, options)  # initialize the optimizer class
       >>> results = asga.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"ASGA: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       ASGA: 5000, 0.0072316975092846245

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/3z8zxr74>`_ for more details.

    Attributes
    ----------
    b                     : `int`
                            number of back-mapped points.
    crossover_prob        : `float`
                            crossover probability.
    mutation_prob         : `float`
                            mutation probability.
    n_individuals         : `int`
                            population size.
    n_initial_individuals : `int`
                            initial population size.
    n_subspace            : `int`
                            dimensionality number of active subspaces.

    References
    ----------
    Demo, N., Tezzele, M. and Rozza, G., 2021.
    A supervised learning approach involving active subspaces for an efficient genetic algorithm in
    high-dimensional optimization problems.
    SIAM Journal on Scientific Computing, 43(3), pp.B831-B853.
    https://epubs.siam.org/doi/10.1137/20M1345219
    """
    def __init__(self, problem, options):
        GA.__init__(self, problem, options)
        self.crossover_prob = options.get('crossover_prob', 0.5)  # crossover probability
        self.mutation_prob = options.get('mutation_prob', 0.5)  # mutation probability
        self.n_subspace = options.get('n_subspace', 1)  # dimensionality number of active subspaces
        self.b = options.get('b', 2)  # number of back-mapped points
        self.alpha = options.get('alpha', 1.0)
        self.n_initial_individuals = options.get('n_initial_individuals', 2000)
        self.n_individuals = options.get('n_individuals', 200)
        self._n_individuals_subspace = int(self.n_individuals/self.b)

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_initial_individuals, self.ndim_problem))  # initial population
        y = np.empty((self.n_initial_individuals,))  # fitness
        for i in range(self.n_initial_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        x_as, y_as = np.copy(x), np.copy(y)
        self._n_generations = 0
        return x, y, x_as, y_as

    def _build_active_space(self, x_as=None, y_as=None):
        active_subspace = ActiveSubspaces(dim=self.n_subspace, method='local')
        active_subspace.fit(inputs=x_as, outputs=y_as)
        return active_subspace

    def _select(self, x=None, y=None):
        return x[np.argsort(y)[:self._n_individuals_subspace]]

    def _crossover(self, x=None):  # different from blend BLX-alpha crossover
        xx = np.copy(x)
        for i in range(self._n_individuals_subspace):
            x1, x2 = self.rng_optimization.choice(x, 2)
            for j in range(self.n_subspace):
                if self.rng_optimization.random() < self.crossover_prob:
                    r = self.rng_optimization.uniform(-1.0*self.alpha, 1.0+self.alpha)
                    xx[i][j] = (1.0 - r)*x1[j] + r*x2[j]
        return xx

    def _mutate(self, x=None):
        for i in range(self._n_individuals_subspace):
            for j in range(self.n_subspace):
                if self.rng_optimization.random() < self.mutation_prob:
                    x[i][j] *= (1 + self.rng_optimization.normal(0, 0.1))
        return x

    def iterate(self, x=None, y=None, x_as=None, y_as=None, args=None):
        active_subspace = self._build_active_space(x_as, y_as)
        xx = self._select(x, y)
        xx = active_subspace.transform(xx)[0]  # forward reduction
        xx = self._mutate(self._crossover(xx))
        x = active_subspace.inverse_transform(xx, self.b)[0]  # backward mapping
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        x_as, y_as = np.vstack((x_as, x)), np.hstack((y_as, y))
        return x, y, x_as, y_as

    def optimize(self, fitness_function=None, args=None):
        fitness = GA.optimize(self, fitness_function)
        x, y, x_as, y_as = self.initialize(args)
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            try:
                x, y, x_as, y_as = self.iterate(x, y, x_as, y_as, args)
                self._n_generations += 1
            except np.linalg.LinAlgError:
                x, y, x_as, y_as = self.initialize(args)
            except ValueError:
                x, y, x_as, y_as = self.initialize(args)
        return self._collect_results(fitness, y)
