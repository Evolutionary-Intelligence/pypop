import numpy as np  # engine for numerical computing
import torch

from lml import LML
from pypop7.optimizers.cem.scem import SCEM


class DCEM(SCEM):
    """Differentiable Cross-Entropy Method (DCEM).

    .. note:: Since the underlying `lml` library may be not successfully installed via `pip`, please run the following
       two commands before invoking this optimizer (**this is a necessary step!**):

       $ `git clone https://github.com/locuslab/lml.git`

       $ `pip install -e lml`

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
                * 'sigma'         - initial global step-size (`float`),
                * 'mean'          - initial mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'n_individuals' - offspring population size (`int`, default: `1000`),
                * 'n_parents'     - parent population size (`int`, default: `200`),
                * 'temperature'   - temperature for lml (`float`, default: `1.0`),
                * 'lml_eps'       - epsilon for lml (`float`, default: `1e-3`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.cem.dcem import DCEM
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 100,
       ...            'lower_boundary': -5*numpy.ones((100,)),
       ...            'upper_boundary': 5*numpy.ones((100,))}
       >>> options = {'max_function_evaluations': 1000000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'sigma': 0.3}  # the global step-size may need to be tuned for better performance
       >>> dcem = DCEM(problem, options)  # initialize the optimizer class
       >>> results = dcem.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"DCEM: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       DCEM: 1000000, 6365.13838155091

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/psd4dxm4>`_ for more details.

    Attributes
    ----------
    lml_eps       : `float`
                    epsilon for lml.
    mean          : `array_like`
                    initial mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.
    sigma         : `float`
                    final global step-size, aka mutation strength (updated during optimization).
    temperature   : `float`
                    temperature for lml.

    References
    ----------
    Amos, B. and Yarats, D., 2020, November.
    The differentiable cross-entropy method.
    In International Conference on Machine Learning (pp. 291-302). PMLR.
    http://proceedings.mlr.press/v119/amos20a.html

    See the official Python code from Amos:
    https://github.com/facebookresearch/dcem
    """
    def __init__(self, problem, options):
        SCEM.__init__(self, problem, options)
        self.temperature = options.get('temperature', 1.0)
        self.lml_eps = options.get('lml_eps', 1e-3)

    def initialize(self, is_restart=False):
        mean = self._initialize_mean(is_restart)
        x = np.empty((self.n_individuals, self.ndim_problem))  # population
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return mean, x, y

    def iterate(self, mean=None, x=None, y=None, args=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            x[i] = self.rng_optimization.normal(mean, self._sigmas)
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y

    def _update_parameters(self, mean=None, x=None, y=None):
        mean_y, std_y = np.mean(y), np.std(y)
        y = torch.from_numpy((y - mean_y)/(std_y + 1e-6))
        i = LML(N=self.n_parents, eps=self.lml_eps, verbose=0)(-y*self.temperature)
        i = i.detach().numpy().reshape(self.n_individuals, 1)
        i_x = i*x
        mean = np.sum(i_x, axis=0)/self.n_parents
        self._sigmas = np.sqrt(np.sum(i*np.power(x - mean, 2), axis=0)/self.n_parents)
        return mean
