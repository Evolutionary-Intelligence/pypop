import numpy as np

from pypop7.optimizers.cem.cem import CEM
import torch
from lml import LML


class DCEM(CEM):
    """Differentiable Cross-Entropy Method

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
              and with the following particular settings (`keys`):
                * 'n_individuals' - population size (`int`, default: `100`),
                * 'n_parents' - parent size (`int`, default: `200`),
                * 'mean' - initial mean value (`array_like`, default: `4 * np.ones((ndim_problem,))`),
                * 'sigma' - initial global step-size (σ), mutation strength (`float`, default: '1.0'),
                * 'lml_verbose' - whether lml model is verbose (`int`, default: `0`),
                * 'lml_eps' - parameter for lml optimization (`float`, default: `0.001`),

    Examples
    --------
    Use the CEM optimizer `DCEM` to minimize the well-known test function
    `Rastrigin <http://en.wikipedia.org/wiki/Rastrigin_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rastrigin  # function to be minimized
       >>> from pypop7.optimizers.cem.dcem import DCEM
       >>> problem = {'fitness_function': rastrigin,  # define problem arguments
       ...            'ndim_problem': 1000,
       ...            'lower_boundary': -5 * numpy.ones((1000,)),
       ...            'upper_boundary': 5 * numpy.ones((1000,))}
       >>> options = {'max_function_evaluations': 1000000,  # set optimizer options
       ...            'n_individuals': 20,
       ...            'n_parents': 10,
       ...            'mean': 4 * np.ones((1000,)),
       ...            'sigma': 1.0,
       ...            'seed_rng': 2022}
       >>> dcem = DCEM(problem, options)  # initialize the optimizer class
       >>> results = decm.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"DCEM: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       DCEM: 1001, 8.731149137020111e-11

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring (λ: lambda), offspring population size.
    n_parents     : `int`
                    number of parents (μ: mu), parental population size.
    mean          : `array_like`
                    mean of Gaussian search distribution.
    sigma         : `float`
                    mutation strength.
    lml_verbose   : `int`
                    whether lml verbose.
    lml_eps       : `float`
                    optimization parameter of lml.

    Reference
    ---------------
    B. Amos, D. Yarats
    The Differentiable Cross-Entropy Method
    Proceedings of the 37th International Conference on Machine Learning, PMLR 119:291-302, 2020.
    http://proceedings.mlr.press/v119/amos20a.html
    """
    def __init__(self, problem, options):
        CEM.__init__(self, problem, options)
        self.lml_verbose = options.get('lml_verbose', 0)
        self.lml_eps = options.get('lml_eps', 0.001)

    def initialize(self, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        mean = self._initialize_mean(is_restart)
        return mean, x, y

    def iterate(self, mean, x, y):
        for i in range(self.n_individuals):
            x[i] = self.rng_optimization.normal(mean, self.sigma, self.ndim_problem)
            x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)
            y[i] = self._evaluate_fitness(x[i])
            if self._check_terminations():
                return x, y
        return x, y

    def update_distribution(self, x, y):
        mean_y = np.mean(y)
        std_y = np.std(y)
        _y = (y - mean_y) / (std_y + 1e-10)
        # _y = _y.reshape(self.n_individuals, 1)
        _y = torch.from_numpy(_y)
        I = LML(N=self.n_parents, verbose=self.lml_verbose, eps=self.lml_eps)(-_y)
        I = I.numpy()
        I = I.reshape(self.n_individuals, 1)
        x_I = I * x
        mean = np.mean(x_I, axis=0)
        self.sigma = np.sqrt(np.mean(I * (x - mean)**2, axis=0))
        return mean

    def optimize(self, fitness_function=None):
        fitness = CEM.optimize(self, fitness_function)
        mean, x, y = self.initialize()
        while True:
            x, y = self.iterate(mean, x, y)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            mean = self.update_distribution(x, y)
        results = self._collect_results(fitness, mean)
        return results
