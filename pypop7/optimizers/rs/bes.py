import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer
from pypop7.optimizers.rs.rs import RS


class BES(RS):
    """BErnoulli Smoothing (BES).

    .. note:: This is a **simplified** version of the recently (2022) proposed BES algorithm **without noisy fitness
       evaluations**. We leave the *noisy fitness evaluations* case for the future work.

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
                * 'n_individuals' - number of individuals/samples (`int`, default: `100`),
                * 'lr'            - learning rate (`float`, default: `0.001`),
                * 'c'             - factor of finite-difference gradient estimate (`float`, default: `0.1`),
                * 'x'                           - initial (starting) point (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

    Examples
    --------
    Use the optimizer `BES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.rs.bes import BES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 100,
       ...            'lower_boundary': -2 * numpy.ones((100,)),
       ...            'upper_boundary': 2 * numpy.ones((100,))}
       >>> options = {'max_function_evaluations': 10000*101,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'n_individuals': 10,
       ...            'c': 0.1,
       ...            'lr': 0.000001}
       >>> bes = BES(problem, options)  # initialize the optimizer class
       >>> results = bes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"BES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       BES: 1010000, 133.79696876596637

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/37hke555>`_ for more details.

    Attributes
    ----------
    c             : `float`
                    factor of finite-difference gradient estimate.
    lr            : `float`
                    learning rate of (estimated) gradient update.
    n_individuals : `int`
                    number of individuals/samples.
    x             : `array_like`
                    initial (starting) point.

    References
    ----------
    Gao, K. and Sener, O., 2022, June.
    Generalizing Gaussian Smoothing for Random Search.
    In International Conference on Machine Learning (pp. 7077-7101). PMLR.
    https://proceedings.mlr.press/v162/gao22f.html
    https://icml.cc/media/icml-2022/Slides/16434.pdf
    """
    def __init__(self, problem, options):
        RS.__init__(self, problem, options)
        self.n_individuals = options.get('n_individuals', 100)  # number of individuals/samples
        self.lr = options.get('lr', 0.001)  # learning rate of (estimated) gradient update
        self.c = options.get('c', 0.1)  # factor of finite-difference gradient estimate
        self.verbose = options.get('verbose', 10)

    def initialize(self, is_restart=False):
        if is_restart or (self.x is None):
            x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        else:
            x = np.copy(self.x)
        y = np.empty((self.n_individuals + 1,))  # no evaluations
        # the fist element of `y` will be used as the base for finite-difference gradient estimate
        return x, y

    def iterate(self, x=None, y=None, args=None):  # for each iteration (generation)
        gradient = np.zeros((self.ndim_problem,))  # estimated gradient
        y[0] = self._evaluate_fitness(x, args)  # for finite-difference gradient estimate
        for i in range(1, self.n_individuals + 1):
            if self._check_terminations():
                return x, y
            # set directional gradient based on Bernoulli distribution
            dg = self.rng_optimization.binomial(n=1, p=0.5, size=(self.ndim_problem,))
            dg = (dg - 0.5)/0.5
            y[i] = self._evaluate_fitness(x + self.c*dg, args)
            gradient += (y[i] - y[0])*dg
        gradient /= (self.c*self.n_individuals)
        x -= self.lr*gradient  # stochastic gradient descent (SGD)
        return x, y

    def optimize(self, fitness_function=None, args=None):  # for all iterations (generations)
        fitness = Optimizer.optimize(self, fitness_function)
        x, y = self.initialize()
        while not self.termination_signal:
            x, y = self.iterate(x, y, args)
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
        return self._collect_results(fitness)
