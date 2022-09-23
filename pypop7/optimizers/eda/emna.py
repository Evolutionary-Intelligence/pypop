import numpy as np

from pypop7.optimizers.eda.umda import UMDA


class EMNA(UMDA):
    """Estimation of Multivariate Normal Algorithm (EMNA).

    .. note:: `EMNA` learns the *full* covariance matrix of the Gaussian sampling distribution, resulting
       in a *cubic* time and space complexity w.r.t. each sampling. Therefore, it is **rarely** used for
       large-scale black-box optimization (LSBBO). It is **highly recommended** to first attempt other more
       advanced methods for LSBBO.

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
    Use the EDA optimizer `EMNA` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.eda.emna import EMNA
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> emna = EMNA(problem, options)  # initialize the optimizer class
       >>> results = emna.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"EMNA: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       EMNA: 5000, 0.008375142194038284

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/2p8xksyy>`_ for more details.

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

    Larranaga, P., Etxeberria, R., Lozano, J.A. and Pena, J.M., 2000.
    Optimization in continuous domains by learning and simulation of Gaussian networks.
    Technical Report, Department of Computer Science and Artificial Intelligence,
    University of the Basque Country.
    https://tinyurl.com/3bw6n3x4
    """
    def __init__(self, problem, options):
        UMDA.__init__(self, problem, options)

    def iterate(self, x=None, y=None, args=None):
        order = np.argsort(y)[:self.n_parents]
        mean = np.mean(x[order], axis=0)
        cov = np.cov(np.transpose(x[order]))
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            x[i] = self.rng_optimization.multivariate_normal(mean, cov)
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y
