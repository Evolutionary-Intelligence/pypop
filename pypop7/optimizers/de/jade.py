import numpy as np
from scipy.stats import cauchy

from pypop7.optimizers.de.de import DE


class JADE(DE):
    """Adaptive Differential Evolution (JADE).

    Parameters
    ----------
    problem : `dict`
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : `dict`
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
                * 'n_individuals' - population size (`int`, default: `100`),
                * 'mu'            - mean of normal distribution for adaptation of crossover probability (`float`,
                  default: `0.5`),
                * 'median'        - median of Cauchy distribution for adaptation of mutation factor (`float`,
                  default: `0.5`),
                * 'p'             - level of greediness of mutation strategy (`float`, default: `0.05`),
                * 'c'             - life span (`float`, default: `0.1`).

    Examples
    --------
    Use the ES optimizer `JADE` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.de.jade import JADE
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 0}
       >>> jade = JADE(problem, options)  # initialize the optimizer class
       >>> results = jade.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"JADE: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       JADE: 5000, 4.844728910084905e-05

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/bddsba5s>`_ for more details.

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring, offspring population size.
    mu            : `float`
                    mean of normal distribution for adaptation of crossover probability.
    median        : `float`
                    median of Cauchy distribution for adaptation of mutation factor.
    p             : `float`
                    level of greediness of mutation strategy.
    c             : `float`
                    life span.

    References
    ----------
    Zhang, J., and Sanderson, A. C. 2009.
    JADE: Adaptive differential evolution with optional external archive.
    IEEE Transactions on Evolutionary Computation, 13(5), pp.945–958.
    https://doi.org/10.1109/TEVC.2009.2014613
    """
    def __init__(self, problem, options):
        DE.__init__(self, problem, options)
        self.mu = options.get('mu', 0.5)  # mean of normal distribution for adaptation of crossover probabilities
        self.median = options.get('median', 0.5)  # location of Cauchy distribution for adaptation of mutation factor
        self.p = options.get('p', 0.05)  # level of greediness of the mutation strategy
        self.c = options.get('c', 0.1)  # life span
        self.archive = options.get('archive', True)
        self.boundary = options.get('boundary', False)

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # population
        y = np.empty((self.n_individuals,))  # fitness
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        a = np.empty((0, self.ndim_problem))  # the set of archived inferior solutions
        self._n_generations += 1
        return x, y, a

    def bound(self, x=None, xx=None):
        if not self.boundary:
            return x
        for k in range(self.n_individuals):
            idx = np.array(x[k] < self.lower_boundary)
            if idx.any():
                x[k][idx] = (self.lower_boundary + xx[k])[idx] / 2
            idx = np.array(x[k] > self.upper_boundary)
            if idx.any():
                x[k][idx] = (self.upper_boundary + xx[k])[idx] / 2
        return x

    def mutate(self, x=None, y=None, a=None):
        x_mu = np.empty((self.n_individuals,  self.ndim_problem))  # mutated population
        f_mu = np.empty((self.n_individuals,))  # mutated mutation factors
        order = np.argsort(y)[:int(np.ceil(self.p*self.n_individuals))]  # index of the 100p% best individuals
        x_p = x[self.rng_optimization.choice(order, (self.n_individuals,))]
        x_un = np.copy(x)
        if self.archive:
            x_un = np.vstack((x_un, a))
        for k in range(self.n_individuals):
            f_mu[k] = cauchy.rvs(loc=self.median, scale=0.1, random_state=self.rng_optimization)
            while f_mu[k] <= 0.0:
                f_mu[k] = cauchy.rvs(loc=self.median, scale=0.1, random_state=self.rng_optimization)
            if f_mu[k] > 1:
                f_mu[k] = 1.0
            r1 = self.rng_optimization.choice(np.setdiff1d(np.arange(self.n_individuals), k))
            r2 = self.rng_optimization.choice(np.setdiff1d(np.arange(len(x_un)), np.union1d(k, r1)))
            x_mu[k] = x[k] + f_mu[k]*(x_p[k] - x[k]) + f_mu[k]*(x[r1] - x_un[r2])
        return x_mu, f_mu

    def crossover(self, x_mu=None, x=None):
        x_cr = np.copy(x)
        p_cr = self.rng_optimization.normal(self.mu, 0.1, (self.n_individuals,))  # crossover probabilities
        p_cr = np.minimum(np.maximum(p_cr, 0.0), 1.0)  # truncate to [0, 1]
        for k in range(self.n_individuals):
            i_rand = self.rng_optimization.integers(self.ndim_problem)
            for i in range(self.ndim_problem):
                if (i == i_rand) or (self.rng_optimization.random() < p_cr[k]):
                    x_cr[k, i] = x_mu[k, i]
        return x_cr, p_cr

    def select(self, args=None, x=None, y=None, x_cr=None, a=None, f_mu=None, p_cr=None):
        f = np.empty((0,))  # the set of all successful mutation factors
        p = np.empty((0,))  # the set of all successful crossover probabilities
        for k in range(self.n_individuals):
            if self._check_terminations():
                break
            yy = self._evaluate_fitness(x_cr[k], args)
            if yy < y[k]:
                a = np.vstack((a, x[k]))  # archive the inferior solution
                f = np.hstack((f, f_mu[k]))  # archive the successful mutation factor
                p = np.hstack((p, p_cr[k]))  # archive the successful crossover probability
                x[k] = x_cr[k]
                y[k] = yy
        if len(p) != 0:
            self.mu = (1 - self.c)*self.mu + self.c*np.mean(p)  # Update mean of normal distribution
        if len(f) != 0:  # Update location of Cauchy distribution
            self.median = (1 - self.c)*self.median + self.c*np.sum(np.power(f, 2))/np.sum(f)
        return x, y, a

    def iterate(self, args=None, x=None, y=None, a=None):
        x_mu, f_mu = self.mutate(x, y, a)
        x_cr, p_cr = self.crossover(x_mu, x)
        x_cr = self.bound(x_cr, x)
        x, y, a = self.select(args, x, y, x_cr, a, f_mu, p_cr)
        if len(a) > self.n_individuals:  # randomly remove solutions from a so that |a| <= self.n_individuals
            a = np.delete(a, self.rng_optimization.choice(len(a), (len(a) - self.n_individuals,), False), 0)
        return x, y, a

    def optimize(self, fitness_function=None, args=None):
        fitness = DE.optimize(self, fitness_function)
        x, y, a = self.initialize(args)
        fitness.extend(y)
        while True:
            x, y, a = self.iterate(args, x, y, a)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        return self._collect_results(fitness)
