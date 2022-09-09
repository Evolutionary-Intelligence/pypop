import numpy as np

from pypop7.optimizers.eda.eda import EDA


class AEMNA(EDA):
    """Adaptive Estimation of Multivariate Normal Algorithm (AEMNA).

    .. note:: `AEMNA` learns the *full* covariance matrix of the Gaussian sampling distribution, resulting
       in *high* time and space complexity in each generation. Therefore, like `EMNA`, it is rarely used
       for large-scale black-box optimization (LSBBO).

       It is **highly recommended** to first attempt other more advanced methods for LSBBO.

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

                * 'verbose'                  - flag to print verbose info during optimization (`bool`, default: `True`),
                * 'verbose_frequency'        - frequency of printing verbose info (`int`, default: `10`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - number of offspring, offspring population size (`int`, default: `200`),
                * 'n_parents'     - number of parents, parental population size (`int`, default:
                  `int(self.n_individuals / 2)`).

    Examples
    --------
    Use the EDA optimizer `AEMNA` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.eda.aemna import AEMNA
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> aemna = AEMNA(problem, options)  # initialize the optimizer class
       >>> results = aemna.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"AEMNA: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       AEMNA: 5000, 0.0023607608362747035

    For its correctness checking of coding, refer to `this code-based repeatability report
    <hhttps://tinyurl.com/5ec2uest>`_ for more details.

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring, offspring population size.
    n_parents     : `int`
                    number of parents, parental population size.

    References
    ----------
    Larrañaga, P. and Lozano, J.A. eds., 2002.
    Estimation of distribution algorithms: A new tool for evolutionary computation.
    Springer Science & Business Media.
    https://link.springer.com/book/10.1007/978-1-4615-1539-5
    """
    def __init__(self, problem, options):
        EDA.__init__(self, problem, options)

    def initialize(self, args=None):
        x = self.rng_optimization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                          size=(self.n_individuals, self.ndim_problem))  # population
        y = np.empty((self.n_individuals,))  # fitness
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        order = np.argsort(y)[:self.n_parents]
        mean, cov = np.mean(x[order], axis=0), np.cov(np.transpose(x[order]))
        return x, y, mean, cov

    def iterate(self, x=None, y=None, mean=None, cov=None, args=None):
        xx = self.rng_optimization.multivariate_normal(mean, cov)
        yy = self._evaluate_fitness(xx, args)
        order = np.argsort(y)[:self.n_parents]
        worst = order[-1]
        if yy < y[worst]:
            mean_bak = np.copy(mean)
            mean += (xx - x[worst])/self.n_parents
            ndim2 = np.power(self.n_parents, 2)
            for i in range(self.ndim_problem):
                for j in range(self.ndim_problem):
                    cov[i, j] = (cov[i, j] - ((xx[i] - x[worst, i])*np.sum(x[order, j] - mean_bak[j]))/ndim2 -
                                 ((xx[j] - x[worst, j])*np.sum(x[order, i] - mean_bak[i]))/ndim2 +
                                 ((xx[i] - x[worst, i])*(xx[j] - x[worst, j]))/ndim2 -
                                 ((x[worst, i] - mean[i])*(x[worst, j] - mean[j]))/self.n_parents +
                                 ((xx[i] - mean[i])*(xx[j] - mean[j]))/self.n_parents)
            x[worst], y[worst] = xx, yy
        return x, y, mean, cov

    def optimize(self, fitness_function=None, args=None):
        fitness = EDA.optimize(self, fitness_function)
        x, y, mean, cov = self.initialize(args)
        fitness.extend(y)
        while True:  # similar to steady-state genetic algorithm
            x, y, mean, cov = self.iterate(x, y, mean, cov, args)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        return self._collect_results(fitness)
