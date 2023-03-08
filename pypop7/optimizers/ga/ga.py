import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class GA(Optimizer):
    """Genetic Algorithm (GA).

    This is the **abstract** class for all `GA` classes. Please use any of its instantiated subclasses to
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
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular setting (`key`):
                * 'n_individuals' - population size (`int`, default: `100`).

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

    De Jong, K.A., 2006.
    Evolutionary computation: A unified approach.
    MIT Press.
    https://mitpress.mit.edu/9780262041942/evolutionary-computation/

    Levine, D., 1997.
    Commentary—Genetic algorithms: A practitioner's view.
    INFORMS Journal on Computing, 9(3), pp.256-259.
    https://pubsonline.informs.org/doi/10.1287/ijoc.9.3.256

    Goldberg, D.E., 1994.
    Genetic and evolutionary algorithms come of age.
    Communications of the ACM, 37(3), pp.113-120.
    https://dl.acm.org/doi/10.1145/175247.175259

    De Jong, K.A., 1993.
    Are genetic algorithms function optimizer?.
    Foundations of Genetic Algorithms, pp.5-17.
    https://www.sciencedirect.com/science/article/pii/B9780080948324500064

    Forrest, S., 1993.
    Genetic algorithms: Principles of natural selection applied to computation.
    Science, 261(5123), pp.872-878.
    https://www.science.org/doi/10.1126/science.8346439

    Mitchell, M., Holland, J. and Forrest, S., 1993.
    When will a genetic algorithm outperform hill climbing.
    Advances in Neural Information Processing Systems (pp. 51-58).
    https://proceedings.neurips.cc/paper/1993/hash/ab88b15733f543179858600245108dd8-Abstract.html

    Holland, J.H., 1992.
    Adaptation in natural and artificial systems: An introductory analysis with applications to
    biology, control, and artificial intelligence.
    MIT press.
    https://direct.mit.edu/books/book/2574/Adaptation-in-Natural-and-Artificial-SystemsAn
    
    Holland, J.H., 1992.
    Genetic algorithms.
    Scientific American, 267(1), pp.66-73.
    https://www.scientificamerican.com/article/genetic-algorithms/

    Goldberg, D.E. and Holland, J.H., 1988.
    Genetic algorithms and machine learning.
    Machine Learning, 3(2), pp.95-99.
    https://link.springer.com/article/10.1023/A:1022602019183

    Holland, J.H., 1973.
    Genetic algorithms and the optimal allocation of trials.
    SIAM Journal on Computing, 2(2), pp.88-105.
    https://epubs.siam.org/doi/10.1137/0202009

    Holland, J.H., 1962.
    Outline for a logical theory of adaptive systems.
    Journal of the ACM, 9(3), pp.297-314.
    https://dl.acm.org/doi/10.1145/321127.321128
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # population size
            self.n_individuals = 100
        assert self.n_individuals > 0
        self._n_generations = 0

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _print_verbose_info(self, fitness, y):
        if self.saving_fitness:
            if not np.isscalar(y):
                fitness.extend(y)
            else:
                fitness.append(y)
        if self.verbose and ((not self._n_generations % self.verbose) or (self.termination_signal > 0)):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect(self, fitness, y=None):
        self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results['_n_generations'] = self._n_generations
        return results
