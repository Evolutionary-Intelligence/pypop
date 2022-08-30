import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class DE(Optimizer):
    """Differential Evolution (DE).

    This is the **base** (abstract) class for all DE classes. Please use any of its concrete subclasses to
    optimize the black-box problem at hand.

    .. note:: Its three methods (`initialize`, `iterate`, `optimize`) should be implemented by its subclasses.

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
              and with one particular setting (`keys`):
                * 'n_individuals' - number of offspring, offspring population size (`int`),

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring, population size.

    Methods
    -------

    References
    ----------
    Storn, R., Price, K. 1997.
    Differential Evolution – A Simple and Efficient Heuristic for global Optimization over Continuous Spaces.
    Journal of Global Optimization, 11, pp.341–359.
    https://doi.org/10.1023/A:1008202821328
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:
            self.n_individuals = 100
        self._n_generations = 0

    def initialize(self):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError

    def crossover(self):
        raise NotImplementedError

    def select(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose_frequency):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect_results(self, fitness=None):
        results = Optimizer._collect_results(self, fitness)
        results['_n_generations'] = self._n_generations
        return results
