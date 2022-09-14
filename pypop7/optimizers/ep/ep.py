import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class EP(Optimizer):
    """Evolutionary Programming (EP).

    This is the **base** (abstract) class for all `EP` classes. Please use any of its instantiated subclasses
    to optimize the black-box problem at hand.

    .. note:: `EP` is one of three classical families of evolutionary algorithms (EAs), proposed originally by Lawrence
    J. Fogel, recipient of both `Evolutionary Computation Pioneer Award 1998 <https://tinyurl.com/456as566>`_ and
    `IEEE Frank Rosenblatt Award 2006 <https://tinyurl.com/yj28zxfa>`_. When used for continuous optimization, most
    of modern `EP` share much similarities (e.g. self-adaptation) with `ES`, another representative EA.

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
                * 'sigma'         - initial global step-size, mutation strength (`float`),
                * 'n_individuals' - number of offspring, offspring population size (`int`, default: `100`).

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring, offspring population size.
    sigma         : `float`
                    initial global step-size, mutation strength.

    Methods
    -------

    References
    ----------
    Yao, X., Liu, Y. and Lin, G., 1999.
    Evolutionary programming made faster.
    IEEE Transactions on Evolutionary Computation, 3(2), pp.82-102.
    https://ieeexplore.ieee.org/abstract/document/771163
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:
            self.n_individuals = 100  # number of offspring, offspring population size
        self.sigma = options.get('sigma')  # initial global step-size, mutation strength
        self._n_generations = 0  # number of generations

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect_results(self, fitness=None):
        results = Optimizer._collect_results(self, fitness)
        results['_n_generations'] = self._n_generations
        return results
