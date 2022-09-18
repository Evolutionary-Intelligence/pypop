import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class GA(Optimizer):
    """Genetic Algorithm (GA).

    This is the **base** (abstract) class for all GA classes. Please use any of its concrete subclasses to
    optimize the black-box problem at hand.

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
                * 'n_individuals' - population size (`int`, default: `100`),

    Attributes
    ----------
    n_individuals : `int`
                    population size.

    Methods
    -------

    References
    ----------
    Whitley, D., 2019.
    Next generation genetic algorithms: A user’s guide and tutorial.
    In Handbook of Metaheuristics (pp. 245-274). Springer, Cham.
    https://link.springer.com/chapter/10.1007/978-3-319-91086-4_8

    Levine, D., 1997.
    Commentary—Genetic algorithms: A practitioner's view.
    INFORMS Journal on Computing, 9(3), pp.256-259.
    https://pubsonline.informs.org/doi/10.1287/ijoc.9.3.256

    Forrest, S., 1996.
    Genetic algorithms.
    ACM Computing Surveys, 28(1), pp.77-80.
    https://dl.acm.org/doi/10.1145/234313.234350

    Goldberg, D.E., 1994.
    Genetic and evolutionary algorithms come of age.
    Communications of the ACM, 37(3), pp.113-120.
    https://dl.acm.org/doi/10.1145/175247.175259

    Forrest, S., 1993.
    Genetic algorithms: Principles of natural selection applied to computation.
    Science, 261(5123), pp.872-878.
    https://www.science.org/doi/10.1126/science.8346439

    De Jong, K.A., 2006.
    Evolutionary computation: A unified approach.
    MIT Press.
    https://mitpress.mit.edu/9780262041942/evolutionary-computation/

    Holland, J.H., 1992.
    Genetic algorithms.
    Scientific American, 267(1), pp.66-73.
    https://www.scientificamerican.com/article/genetic-algorithms/

    Goldberg, D.E. and Holland, J.H., 1988.
    Genetic algorithms and machine learning.
    Machine Learning, 3(2), pp.95-99.
    https://link.springer.com/article/10.1023/A:1022602019183
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # population size
            self.n_individuals = 100
        self._n_generations = 0

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect_results(self, fitness):
        results = Optimizer._collect_results(self, fitness)
        results['_n_generations'] = self._n_generations
        return results
