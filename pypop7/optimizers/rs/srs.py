import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer
from pypop7.optimizers.rs.prs import PRS


class SRS(PRS):
    """Simple Random Search (SRS).

    .. note:: `SRS` is an *adaptive* random search method, originally designed by Rosenstein and `Barto
       <https://people.cs.umass.edu/~barto/>`_ for **direct policy search** in reinforcement learning.
       Since it uses the simple *individual-based* random sampling strategy, it easily suffers from a
       *limited* exploration ability for large-scale black-box optimization (LSBBO). Therefore, it is
       **highly recommended** to first attempt other more advanced (e.g. population-based) methods for
       LSBBO. Here we include it only for *benchmarking* purpose.

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
                * 'x'         - initial (starting) point (`array_like`),
                * 'sigma'     - initial global step-size (`float`),
                * 'alpha'     - factor of global step-size (`float`, default: `0.3`),
                * 'beta'      - adjustment probability for exploration-exploitation trade-off (`float`, default: `0`),
                * 'gamma'     - factor of search decay (`float`, default: `0.99`),
                * 'min_sigma' - minimum of global step-size (`float`, default: `0.01`).

    For its correctness checking of coding, the code-based repeatability report cannot be provided owing to
    the lack of its simulation envrionment.

    Examples
    --------
    Use the Random Search optimizer `SRS` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.rs.srs import SRS
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3 * numpy.ones((2,)),
       ...            'sigma': 0.1}
       >>> srs = SRS(problem, options)  # initialize the optimizer class
       >>> results = srs.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"SRS: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       SRS: 5000, 0.0017821578376762473

    Attributes
    ----------
    alpha                       : `float`
                                  factor of global step-size.
    beta                        : `float`
                                  adjustment probability for exploration-exploitation trade-off.
    gamma                       : `float`
                                  factor of search decay.
    min_sigma                   :  `float`
                                  minimum of global step-size.
    sigma                       : `float`
                                  global step-size.
    x                           : `array_like`
                                  initial (starting) point.

    References
    ----------
    Rosenstein, M.T. and Grupen, R.A., 2002, May.
    Velocity-dependent dynamic manipulability.
    In Proceedings of IEEE International Conference on Robotics and Automation (pp. 2424-2429). IEEE.
    https://ieeexplore.ieee.org/abstract/document/1013595

    Rosenstein, M.T. and Barto, A.G., 2001, August.
    Robot weightlifting by direct policy search.
    In International Joint Conference on Artificial Intelligence (pp. 839-846).
    https://dl.acm.org/doi/abs/10.5555/1642194.1642206
    """
    def __init__(self, problem, options):
        # only support normally-distributed random sampling during optimization
        options['sampling_distribution'] = 0
        PRS.__init__(self, problem, options)
        self.alpha = options.get('alpha', 0.3)  # factor of (global) step-size
        assert self.alpha > 0.0, f'`self.alpha` == {self.alpha}, but should > 0.0.'
        self.beta = options.get('beta', 0.0)  # adjustment probability for exploration-exploitation trade-off
        assert 0.0 <= self.beta <= 1.0, f'`self.beta` == {self.beta}, but should >= 0.0 and <= 1.0.'
        self.gamma = options.get('gamma', 0.99)  # factor of search decay
        assert 0.0 <= self.gamma <= 1.0, f'`self.gamma` == {self.gamma}, but should >= 0.0 and <= 1.0.'
        self.min_sigma = options.get('min_sigma', 0.01)  # minimum of (global) step-size
        assert self.min_sigma > 0.0, f'`self.min_sigma` == {self.min_sigma}, but should > 0.0.'

    def initialize(self, args=None):
        if self.x is None:
            x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        else:
            x = np.copy(self.x)
        y = self._evaluate_fitness(x, args)
        return x, y

    def iterate(self, x=None, args=None):
        delta_x = self.sigma*self.rng_optimization.standard_normal(size=(self.ndim_problem,))
        y = self._evaluate_fitness(x + delta_x, args)  # random perturbation
        if self.rng_optimization.uniform() < self.beta:
            x += self.alpha*delta_x
        else:
            x += self.alpha*(self.best_so_far_x - x)
        self._n_generations += 1
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        x, y = self.initialize(args)
        if self.saving_fitness:
            fitness.append(y)
        self._print_verbose_info(y)
        while True:
            x, y = self.iterate(x, args)
            if self.saving_fitness:
                fitness.append(y)
            self._print_verbose_info(y)
            if self._check_terminations():
                break
            self.sigma = np.maximum(self.gamma*self.sigma, self.min_sigma)
        return self._collect_results(fitness)
