import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class DE(Optimizer):
    """Differential Evolution (DE).

    This is the **base** (abstract) class for all `DE` classes. Please use any of its instantiated subclasses
    to optimize the black-box problem at hand.

    .. note:: Originally `DE` was proposed to solve some challenging real-world black-box problems by Kenneth Price and
       Rainer Storn, `recipients of Evolutionary Computation Pioneer Award 2017 <https://tinyurl.com/456as566>`_.
       Although there are *few* significant theoretical advances till now (to our knowledge), it is **still widely used
       in practice**, owing to its often attractive search performance on many multimodal black-box functions.

       The popular and powerful `SciPy <https://www.nature.com/articles/s41592-019-0686-2>`_ library has provided an
       open-source Python implementation for `DE`.

       `"DE borrows the idea from Nelder&Mead of employing information from within the vector population to alter
       the search space." ---[Storn&Price, 1997, JGO] <https://doi.org/10.1023/A:1008202821328>`_

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
              and with the following particular setting (`key`):
                * 'n_individuals' - number of offspring, offspring population size (`int`, default: `100`).

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring, offspring population size. For `DE`, typically a *large* (often >=100)
                    population size is used to better explore for multimodal functions. Obviously the *optimal*
                    population size is problem-dependent, relying on fine-tuning.

    Methods
    -------

    References
    ----------
    Price, K.V., 2013.
    Differential evolution.
    In Handbook of Optimization (pp. 187-214). Springer, Berlin, Heidelberg.
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
        self._n_generations = 0  # number of generations

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
        if self.verbose and (not self._n_generations % self.verbose):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect_results(self, fitness=None):
        results = Optimizer._collect_results(self, fitness)
        results['_n_generations'] = self._n_generations
        return results
