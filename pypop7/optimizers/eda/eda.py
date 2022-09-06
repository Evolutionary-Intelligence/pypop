import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class EDA(Optimizer):
    """Estimation of Distribution Algorithms (EDA).

    This is the **base** (abstract) class for all EDA classes. Please use any of its concrete subclasses to
    optimize the black-box problem at hand.

    .. note:: Its three methods (`initialize`, `iterate`, `optimize`) should be implemented by all its
       subclasses. *`EDA` are a modern branch of evolutionary algorithms with some unique advantages in
       principle*, as recognized in `[Kabán et al., 2016, ECJ] <https://tinyurl.com/mrxpe28y>`_.

       AKA probabilistic model-building genetic algorithms (PMBGA), iterated density estimation
       evolutionary algorithms (IDEA).

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
              and with the following particular settings (`keys`):
                * 'n_individuals' - number of offspring, offspring population size (`int`, default: `200`),
                * 'n_parents'     - number of parents, parental population size (`int`, default:
                  `int(self.n_individuals / 2)`).

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring, offspring population size.
    n_parents     : `int`
                    number of parents, parental population size.

    Methods
    -------

    References
    ----------
    https://www.dagstuhl.de/en/program/calendar/semhp/?semnr=22182

    Larrañaga, P. and Lozano, J.A. eds., 2001.
    Estimation of distribution algorithms: A new tool for evolutionary computation.
    Springer Science & Business Media.
    https://link.springer.com/book/10.1007/978-1-4615-1539-5

    Mühlenbein, H. and Mahnig, T., 2001.
    Evolutionary algorithms: From recombination to search distributions.
    In Theoretical Aspects of Evolutionary Computing (pp. 135-173). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-662-04448-3_7

    Berny, A., 2000, September.
    Selection and reinforcement learning for combinatorial optimization.
    In International Conference on Parallel Problem Solving from Nature (pp. 601-610).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/3-540-45356-3_59

    Bosman, P.A. and Thierens, D., 2000, September.
    Expanding from discrete to continuous estimation of distribution algorithms: The IDEA.
    In International Conference on Parallel Problem Solving from Nature (pp. 767-776).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/3-540-45356-3_75

    Mühlenbein, H., 1997.
    The equation for response to selection and its use for prediction.
    Evolutionary Computation, 5(3), pp.303-346.
    https://tinyurl.com/yt78c786
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # number of offspring, offspring population size
            self.n_individuals = 200
        if self.n_parents is None:  # number of parents, parental population size
            self.n_parents = int(self.n_individuals / 2)
        self.n_restart = 0
        self._n_generations = 0

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose_frequency):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y,
                              np.min(y), self.n_function_evaluations))

    def _collect_results(self, fitness):
        results = Optimizer._collect_results(self, fitness)
        results['_n_generations'] = self._n_generations
        results['n_restart'] = self.n_restart
        return results
