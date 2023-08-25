import numpy as np
from scipy.stats import multivariate_normal

from pypop7.optimizers.eda.emna import EMNA
from pypop7.optimizers.eda.eda import EDA


class EMNAWA(EMNA):
    """Estimation of Multivariate Normal Algorithm with Weighted Averages (EMNAWA).

    .. note:: .

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
                * 'n_individuals' - number of offspring, offspring population size (`int`, default: `200`),
                * 'n_parents'     - number of parents, parental population size (`int`, default:
                  `int(options['n_individuals']/2)`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.eda.emnawa import EMNAWA
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> emnawa = EMNAWA(problem, options)  # initialize the optimizer class
       >>> results = emnawa.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"EMNAWA: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       EMNAWA: 5000, 0.008375142194038284

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/2p8xksyy>`_ for more details.

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.

    References
    ----------
    Teytaud, F. and Teytaud, O., 2009, July.
    Why one must use reweighting in estimation of distribution algorithms.
    In Proceedings of ACM Annual Conference on Genetic and Evolutionary Computation (pp. 453-460)
    https://doi.org/10.1145/1569901.1569964
    """
    def __init__(self, problem, options):
        EMNA.__init__(self, problem, options)

    def initialize(self, args=None):
        mean, cov = (self.initial_upper_boundary + self.initial_lower_boundary) / 2, 0.1 * np.eye(self.ndim_problem)
        x = self.rng_optimization.multivariate_normal(mean, cov, size=(self.n_individuals,))  # population
        y = np.empty((self.n_individuals,))  # fitness
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y, mean, cov

    def iterate(self, x=None, y=None, mean=None, cov=None, args=None):
        order = np.argsort(y)[:self.n_parents]
        try:
            m = multivariate_normal(mean=mean, cov=cov)
        except Exception:
            m = multivariate_normal(mean=mean, cov=cov + 1e-100 * np.eye(self.ndim_problem))

        w = 1 / m.pdf(x[order]).reshape(-1, 1)
        w = w / np.sum(w)
        x[order] += (x[order] - mean)*w
        mean = np.mean(x[order], axis=0)
        cov = np.cov(np.transpose(x[order]))
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            x[i] = self.rng_optimization.multivariate_normal(mean, cov)
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y, mean, cov

    def optimize(self, fitness_function=None, args=None):
        fitness = EDA.optimize(self, fitness_function)
        x, y, mean, cov = self.initialize(args)
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            x, y, mean, cov = self.iterate(x, y, mean, cov, args)
            self._n_generations += 1
        return self._collect(fitness, y)
