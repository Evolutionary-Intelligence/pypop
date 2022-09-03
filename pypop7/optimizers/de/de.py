import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class DE(Optimizer):
    """Differential Evolution (DE).

    This is the **base** (abstract) class for all DE classes. Please use any of its concrete subclasses to
    optimize the black-box problem at hand.

    .. note:: Its six methods (`initialize`, `mutate`, `crossover`, `select`, `iterate`, `optimize`) should
       be implemented by its subclasses.

       Originally `DE` was proposed to solve some challenging real-world problems by Kenneth Price and Rainer Storn,
       `recipients of Evolutionary Computation Pioneer Award 2017 <https://tinyurl.com/456as566>`_. Although there
       are few significant theoretical advances, it is still widely used in practice owing to its often attractive
       performance on multimodal functions (`SciPy <https://www.nature.com/articles/s41592-019-0686-2>`_ has provided
       an open-source implementation for `DE`).

       *"DE borrows the idea from Nelder&Mead of employing information from within the vector population to alter
       the search space."* --- `Storn&Price, 1997, JGO <https://doi.org/10.1023/A:1008202821328>`_

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
                * 'n_individuals' - number of offspring, offspring population size (`int`).

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring, offspring population size.

    Methods
    -------

    References
    ----------
    Price, K.V., 2013.
    Differential evolution.
    In Handbook of optimization (pp. 187-214). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-642-30504-7_8

    Price, K.V., Storn, R.M. and Lampinen, J.A., 2005.
    Differential evolution: A practical approach to global optimization.
    Springer Science & Business Media.
    https://link.springer.com/book/10.1007/3-540-31306-0

    Storn, R.M. and Price, K.V. 1997.
    Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces.
    Journal of Global Optimization, 11(4), pp.341–359.
    https://doi.org/10.1023/A:1008202821328
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # number of offspring, offspring population size
            self.n_individuals = 100
        self._n_generations = 0

    def initialize(self):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError

    def crossover(self):
        raise NotImplementedError

    def bound(self):
        pass

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
