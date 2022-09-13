import numpy as np

from pypop7.optimizers.rs.rs import RS


class PRS(RS):
    """Pure Random Search (PRS).

    .. note:: `PRS` is one of the *simplest* and *earliest* black-box optimizers. Although recently it
        has been successfully applied in several *relatively low-dimensional* problems (particularly
        `hyper-parameter optimization <https://www.jmlr.org/papers/v13/bergstra12a.html>`_), it generally
        suffers from the famous **curse of dimensionality** for large-scale black-box optimization (LSBBO),
        since its lack of *adaptation*, a highly desirable property for sophisticated search algorithms. It
        is **highly recommended** to first attempt other more advanced (e.g. population-based) methods for LSBBO.

        Here we include it mainly for *benchmarking* purpose.

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

                * 'verbose'                  - flag to print verbose information during optimization (`bool`, default:
                  `True`),
                * 'verbose_frequency'        - generation frequency of printing verbose information (`int`, default:
                  `1000`).

    Examples
    --------
    Use the Random Search optimizer `PRS` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.rs.prs import PRS
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3 * numpy.ones((2,))}
       >>> prs = PRS(problem, options)  # initialize the optimizer class
       >>> results = prs.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"PRS: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       PRS: 5000, 0.11497678820610932

    References
    ----------
    Bergstra, J. and Bengio, Y., 2012.
    Random search for hyper-parameter optimization.
    Journal of Machine Learning Research, 13(2).
    https://www.jmlr.org/papers/v13/bergstra12a.html

    Brooks, S.H., 1958.
    A discussion of random methods for seeking maxima.
    Operations Research, 6(2), pp.244-251.
    https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244
    """
    def __init__(self, problem, options):
        RS.__init__(self, problem, options)
        # default: 1 -> uniformly distributed random sampling
        self.sampling_distribution = options.get('sampling_distribution', 1)
        if self.sampling_distribution not in [0, 1]:  # 0 -> normally distributed random sampling
            info = 'For {:s}, only support uniformly or normally distributed random sampling.'
            raise ValueError(info.format(self.__class__.__name__))
        if self.sampling_distribution == 0:
            self.sigma = options.get('sigma')  # initial (global) step-size
            if self.sigma is None:
                raise ValueError('`sigma` should be set.')

    def _sample(self, rng):
        if self.sampling_distribution == 0:
            x = self.x + self.sigma*rng.standard_normal(size=(self.ndim_problem,))
        else:
            x = rng.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        return x

    def initialize(self):
        if self.x is None:
            x = self._sample(self.rng_initialization)
        else:
            x = np.copy(self.x)
        return x

    def iterate(self):  # draw a sample (individual)
        return self._sample(self.rng_optimization)
