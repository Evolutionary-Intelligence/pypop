import numpy as np

from pypop7.optimizers.cc.cc import CC


class COSYNE(CC):
    """CoOperative SYnapse NEuroevolution (COSYNE).

    .. note:: This is a wrapper of `COSYNE`, which has been implemented in the Python library `EvoTorch
       <https://docs.evotorch.ai/v0.3.0/reference/evotorch/algorithms/ga/#evotorch.algorithms.ga.Cosyne>`_,
       with slight modifications.

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
                * 'sigma'          - initial global step-size for Gaussian search distribution (`float`),
                * 'n_individuals'  - number of individuals/samples, aka population size (`int`, default: `100`),
                * 'n_tournaments'  - number of tournaments for one-point crossover (`int`, default: `10`),
                * 'ratio_elitists' - ratio of elitists (`float`, default: `0.3`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.cc.cosyne import COSYNE
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'sigma': 0.3,
       ...            'x': 3*numpy.ones((2,))}
       >>> cosyne = COSYNE(problem, options)  # initialize the optimizer class
       >>> results = cosyne.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"COSYNE: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       COSYNE: 5000, 0.005023488269997175

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/yff8c6xu>`_ for more details.

    Attributes
    ----------
    n_individuals  : `int`
                     number of individuals/samples, aka population size.
    n_tournaments  : `int`
                     number of tournaments for one-point crossover.
    ratio_elitists : `float`
                     ratio of elitists.
    sigma          : `float`
                     initial global step-size for Gaussian search (mutation/sampling) distribution.

    References
    ----------
    Gomez, F., Schmidhuber, J. and Miikkulainen, R., 2008.
    Accelerated neural evolution through cooperatively coevolved synapses.
    Journal of Machine Learning Research, 9(31), pp.937-965.
    https://jmlr.org/papers/v9/gomez08a.html

    https://docs.evotorch.ai/v0.3.0/reference/evotorch/algorithms/ga/#evotorch.algorithms.ga.Cosyne
    https://github.com/nnaisense/evotorch/blob/master/src/evotorch/algorithms/ga.py
    """
    def __init__(self, problem, options):
        CC.__init__(self, problem, options)
        self.sigma = options.get('sigma')  # global step-size for Gaussian search distribution
        self.n_tournaments = options.get('n_tournament', 10)  # number of tournaments for one-point crossover
        self.ratio_elitists = options.get('ratio_elitists', 0.3)  # ratio of elitists
        self._n_elitists = int(self.ratio_elitists*self.n_individuals)  # number of elitists
        self._n_parents = int(self.n_individuals/4)  # parents for crossover and mutation

    def initialize(self, args=None, is_restart=False):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # population
        y = np.empty((self.n_individuals,))  # fitness
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y

    def _crossover(self, x, y):  # one-point crossover
        xx = np.empty((2*len(y), self.ndim_problem))
        for i in range(len(y)):
            left = self.rng_optimization.choice(len(y), size=(self.n_tournaments,), replace=False)
            left = x[np.argmin(y[left])]
            right = self.rng_optimization.choice(len(y), size=(self.n_tournaments,), replace=False)
            right = x[np.argmin(y[right])]
            p = self.rng_optimization.choice(self.ndim_problem)
            xx[2*i], xx[2*i + 1] = np.append(left[:p], right[p:]), np.append(right[:p], left[p:])
        return xx

    def _mutate(self, x):  # mutation for all dimensions
        x += self.sigma*self.rng_optimization.standard_normal(size=(x.shape[0], self.ndim_problem))
        x = np.clip(x, self.lower_boundary, self.upper_boundary)
        return x

    def _permute(self, x):  # different from the original paper for simplicity
        xx = np.copy(x)
        for d in range(self.ndim_problem):
            p = self.rng_optimization.choice(self.n_individuals)
            xx[:, d] = np.append(xx[p:, d], xx[:p, d])
        return xx

    def iterate(self, x=None, y=None, args=None):
        order, yy, yyy = np.argsort(y), np.empty((2*self._n_parents,)), np.empty((self.n_individuals,))
        xx = self._mutate(self._crossover(x[order[:self.n_parents]], y[order[:self.n_parents]]))
        for i in range(2*self._n_parents):
            if self._check_terminations():
                return x, y, np.append(yy, yyy)
            yy[i] = self._evaluate_fitness(xx[i], args)
        xxx = self._permute(x)
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y, np.append(yy, yyy)
            yyy[i] = self._evaluate_fitness(xxx[i], args)
        x = np.vstack((np.vstack((x[order[:self._n_elitists]], xx)), xxx))
        y = np.hstack((np.hstack((y[order[:self._n_elitists]], yy)), yyy))
        order = np.argsort(y)[:self.n_individuals]  # to keep population size fixed
        self._n_generations += 1
        return x[order], y[order], np.append(yy, yyy)

    def optimize(self, fitness_function=None, args=None):
        fitness = CC.optimize(self, fitness_function)
        x, y = self.initialize(args)
        yy = y  # only for printing
        while not self._check_terminations():
            self._print_verbose_info(fitness, yy)
            x, y, yy = self.iterate(x, y, args)
        return self._collect(fitness, yy)
