import numpy as np

from pypop7.optimizers.eda.eda import EDA


class RPEDA(EDA):
    """Random-Projection Estimation of Distribution Algorithm (RPEDA).

    .. note:: `RPEDA` uses **random matrix theory (RMT)** to sample individuals on multiple embedded subspace,
       though it still evaluates individuals on the original search space. It has a **quadractic** time
       complexity w.r.t. each sampling for large-scale black-box optimization.

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
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - number of offspring, offspring population size (`int`, default: `300`),
                * 'n_parents'     - number of parents, parental population size (`int`, default:
                  `int(0.25*self.n_individuals)`),
                * 'k'             - projection dimensionality (`int`, default: `3`),
                * 'm'             - number of random projection matrices (`int`, default:
                  `int(np.ceil(4*self.ndim_problem/self.k)`).

    Examples
    --------
    Use the EDA optimizer `RPEDA` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.eda.rpeda import RPEDA
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'k': 2}
       >>> rpeda = RPEDA(problem, options)  # initialize the optimizer class
       >>> results = rpeda.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"RPEDA: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       RPEDA: 5000, 0.05347416897341553

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring, offspring population size.
    n_parents     : `int`
                    number of parents, parental population size.
    k             : `int`
                    projection dimensionality.
    m             : `int`
                    number of random projection matrices.

    References
    ----------
    Kabán, A., Bootkrajang, J. and Durrant, R.J., 2016.
    Toward large-scale continuous EDA: A random matrix theory perspective.
    Evolutionary Computation, 24(2), pp.255-291.
    https://direct.mit.edu/evco/article-abstract/24/2/255/1016/Toward-Large-Scale-Continuous-EDA-A-Random-Matrix
    """
    def __init__(self, problem, options):
        EDA.__init__(self, problem, options)
        self.n_individuals = options.get('n_individuals', 300)  # population size
        n_parents = int(0.25*self.n_individuals)
        self.n_parents = options.get('n_parents', n_parents)  # number of selected individuals
        self.k = options.get('k', 3)  # projection dimensionality
        assert self.k < self.ndim_problem
        size_rpm = int(np.ceil(4*self.ndim_problem/self.k))
        self.m = options.get('m', size_rpm)  # number of random projection matrices
        self._sq_1_d = np.sqrt(1.0/self.ndim_problem)
        self._sq_nsk = np.sqrt(self.ndim_problem*self.m/self.k)

    def initialize(self):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # population
        xx = np.copy(x)  # top individuals
        y = np.empty((self.n_individuals,))  # fitness
        return x, xx, y

    def iterate(self, xx=None, y=None, args=None):
        mean = np.mean(xx, axis=0)
        diff = xx - mean  # centred points to be projected
        x = np.zeros((self.n_individuals, self.ndim_problem))
        for i in range(self.m):
            # project into a k-dimensional subspace
            r = self.rng_optimization.standard_normal((self.ndim_problem, self.k))*self._sq_1_d
            pm = np.dot(diff, r)  # projection
            cm = np.cov(np.transpose(pm))  # covariance matrix
            samples = self.rng_optimization.multivariate_normal(np.zeros((self.k,)), cm,
                                                                size=(self.n_individuals,))
            # recover into the original space
            x += np.dot(samples, np.transpose(r))
        x /= self.m  # averaging
        x *= self._sq_nsk  # scaling
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            if i == 0:  # elitism
                x[0] = xx[0]
            else:
                x[i] = np.clip(x[i] + mean, self.lower_boundary, self.upper_boundary)
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = EDA.optimize(self, fitness_function)
        x, xx, y = self.initialize()
        while True:
            x, y = self.iterate(xx, y, args)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            # select top individuals
            order = np.argsort(y)[:self.n_parents]
            xx = np.copy(x[order])
            self._print_verbose_info(y)
            self._n_generations += 1
        return self._collect_results(fitness)
