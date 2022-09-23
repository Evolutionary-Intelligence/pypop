import numpy as np

from pypop7.optimizers.eda.eda import EDA


class UMDA(EDA):
    """Univariate Marginal Distribution Algorithm for normal models (UMDA).

    .. note:: `UMDA` learns only the *diagonal* elements of covariance matrix of the Gaussian sampling
       distribution, resulting in a *linear* time complexity w.r.t. each sampling. Therefore, it can be
       seen as a *baseline* for large-scale black-box optimization (LSBBO). To obtain satisfactory
       performance for LSBBO, the number of offspring may need to be carefully tuned.

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
                  `int(self.n_individuals/2)`).

    Examples
    --------
    Use the EDA optimizer `UMDA` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.eda.umda import UMDA
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> umda = UMDA(problem, options)  # initialize the optimizer class
       >>> results = umda.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"UMDA: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       UMDA: 5000, 0.029323401402499186

    For its correctness checking, refer to `this code-based repeatability report
    <https://tinyurl.com/2p8m78r2>`_ for more details.

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring, offspring population size.
    n_parents     : `int`
                    number of parents, parental population size.

    References
    ----------
    Larrañaga, P. and Lozano, J.A. eds., 2001.
    Estimation of distribution algorithms: A new tool for evolutionary computation.
    Springer Science & Business Media.
    https://link.springer.com/book/10.1007/978-1-4615-1539-5

    Mühlenbein, H. and Mahnig, T., 2001.
    Evolutionary algorithms: From recombination to search distributions.
    In Theoretical Aspects of Evolutionary Computing (pp. 135-173). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-662-04448-3_7

    Larranaga, P., Etxeberria, R., Lozano, J.A. and Pena, J.M., 2000.
    Optimization in continuous domains by learning and simulation of Gaussian networks.
    Technical Report, Department of Computer Science and Artificial Intelligence,
    University of the Basque Country.
    https://tinyurl.com/3bw6n3x4

    Larranaga, P., Etxeberria, R., Lozano, J.A. and Pe, J.M., 1999.
    Optimization by learning and simulation of Bayesian and Gaussian networks.
    Technical Report, Department of Computer Science and Artificial Intelligence,
    University of the Basque Country.
    https://tinyurl.com/5dktrdwc

    Mühlenbein, H., 1997.
    The equation for response to selection and its use for prediction.
    Evolutionary Computation, 5(3), pp.303-346.
    https://tinyurl.com/yt78c786
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
        return x, y

    def iterate(self, x=None, y=None, args=None):
        order = np.argsort(y)[:self.n_parents]
        mean, sigmas = np.mean(x[order], axis=0), np.std(x[order], axis=0)
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            x[i] = mean + sigmas*self.rng_optimization.standard_normal(size=(self.ndim_problem,))
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = EDA.optimize(self, fitness_function)
        x, y = self.initialize(args)
        if self.saving_fitness:
            fitness.extend(y)
        self._print_verbose_info(y)
        while True:
            x, y = self.iterate(x, y, args)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        return self._collect_results(fitness)
