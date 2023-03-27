import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class EDA(Optimizer):
    """Estimation of Distribution Algorithms (EDA).

    This is the **abstract** class for all `EDA` classes. Please use any of its instantiated subclasses to
    optimize the black-box problem at hand.

    .. note:: *`EDA` are a modern branch of evolutionary algorithms with some unique advantages in
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
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default: `200`),
                * 'n_parents'     - number of parents, aka parental population size (`int`, default:
                  `int(self.n_individuals/2)`).

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.

    Methods
    -------

    References
    ----------
    https://www.dagstuhl.de/en/program/calendar/semhp/?semnr=22182

    Brookes, D., Busia, A., Fannjiang, C., Murphy, K. and Listgarten, J., 2020, July.
    A view of estimation of distribution algorithms through the lens of expectation-maximization.
    In Proceedings of Genetic and Evolutionary Computation Conference Companion (pp. 189-190). ACM.
    https://dl.acm.org/doi/abs/10.1145/3377929.3389938

    Kabán, A., Bootkrajang, J. and Durrant, R.J., 2016.
    Toward large-scale continuous EDA: A random matrix theory perspective.
    Evolutionary Computation, 24(2), pp.255-291.
    https://direct.mit.edu/evco/article-abstract/24/2/255/1016/Toward-Large-Scale-Continuous-EDA-A-Random-Matrix

    Larrañaga, P. and Lozano, J.A. eds., 2002.
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

    Baluja, S. and Caruana, R., 1995.
    Removing the genetics from the standard genetic algorithm.
    In International Conference on Machine Learning (pp. 38-46). Morgan Kaufmann.
    https://www.sciencedirect.com/science/article/pii/B9781558603776500141
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # number of offspring, aka offspring population size
            self.n_individuals = 200
        if self.n_parents is None:  # number of parents, aka parental population size
            self.n_parents = int(self.n_individuals/2)
        self._n_generations = 0
        self._n_restart = 0
        self._printed_evaluations = self.n_function_evaluations

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _print_verbose_info(self, fitness, y, is_print=False):
        if y is not None and self.saving_fitness:
            if not np.isscalar(y):
                fitness.extend(y)
            else:
                fitness.append(y)
        if self.verbose:
            is_verbose = self._printed_evaluations != self.n_function_evaluations  # to avoid repeated printing
            is_verbose_1 = (not self._n_generations % self.verbose) and is_verbose
            is_verbose_2 = self.termination_signal > 0 and is_verbose
            is_verbose_3 = is_print and is_verbose
            if is_verbose_1 or is_verbose_2 or is_verbose_3:
                info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
                print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))
                self._printed_evaluations = self.n_function_evaluations

    def _collect(self, fitness=None, y=None):
        self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results['_n_generations'] = self._n_generations
        results['_n_restart'] = self._n_restart
        return results
