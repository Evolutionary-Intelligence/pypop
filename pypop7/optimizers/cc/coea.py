import time

import numpy as np

from pypop7.optimizers.ga.genitor import GENITOR
from pypop7.optimizers.cc import CC


class COEA(CC):
    """CoOperative co-Evolutionary Algorithm (COEA).

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
                * 'n_individuals'  - number of individuals/samples, aka population size (`int`, default: `100`).

    Examples
    --------
    Use the optimizer `COEA` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.cc.coea import COEA
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'sigma': 0.3,
       ...            'x': 3 * numpy.ones((2,))}
       >>> coea = COEA(problem, options)  # initialize the optimizer class
       >>> results = coea.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"COEA: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       COEA: 5000, 0.8300976710280922

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/ap7n389u>`_ for more details.

    Attributes
    ----------
    n_individuals  : `int`
                     number of individuals/samples, aka population size.

    References
    ----------
    Potter, M.A. and De Jong, K.A., 1994, October.
    A cooperative coevolutionary approach to function optimization.
    In International Conference on Parallel Problem Solving from Nature (pp. 249-257).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/3-540-58484-6_269
    """
    def __init__(self, problem, options):
        CC.__init__(self, problem, options)

    def initialize(self, arg=None):
        sub_optimizers = []
        for i in range(self.ndim_problem):
            problem = {'ndim_problem': 1,
                       'lower_boundary': self.lower_boundary[i],
                       'upper_boundary': self.upper_boundary[i]}
            options = {'seed_rng': self.rng_optimization.integers(np.iinfo(np.int64).max),
                       'n_individuals': self.n_individuals,
                       'max_runtime': self.max_runtime,
                       'verbose': False}
            genitor = GENITOR(problem, options)
            genitor.start_time = time.time()
            sub_optimizers.append(genitor)
        self.best_so_far_x = self.rng_optimization.uniform(self.lower_boundary, self.upper_boundary)
        self.best_so_far_y = self._evaluate_fitness(self.best_so_far_x, arg)
        return sub_optimizers, self.best_so_far_y

    def iterate(self):
        pass

    def optimize(self, fitness_function=None, args=None):
        fitness, is_initialization = CC.optimize(self, fitness_function), True
        sub_optimizers, y = self.initialize(args)
        x_s, yy_s, cop_s, yy = [], [], [], []
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            y = []
            if is_initialization:
                is_initialization = False
                for i, opt in enumerate(sub_optimizers):
                    if self._check_terminations():
                        break

                    def sub_function(sub_x):  # to define sub-function for each sub-optimizer
                        best_so_far_x = np.copy(self.best_so_far_x)
                        best_so_far_x[i] = sub_x
                        return self._evaluate_fitness(best_so_far_x, args)
                    opt.fitness_function = sub_function
                    x, yy, crossover_probs = opt.initialize()
                    x_s.append(x)
                    yy_s.append(yy)
                    cop_s.append(crossover_probs)
                    y.extend(yy)
            else:
                for i, opt in enumerate(sub_optimizers):
                    if self._check_terminations():
                        break

                    def sub_function(sub_x):  # to define sub-function for each sub-optimizer
                        best_so_far_x = np.copy(self.best_so_far_x)
                        best_so_far_x[i] = sub_x
                        return self._evaluate_fitness(best_so_far_x, args)
                    opt.fitness_function = sub_function
                    _, yy, _ = opt.iterate(x_s[i], yy_s[i], cop_s[i])
                    y.append(yy)
            self._n_generations += 1
        return self._collect_results(fitness, y)
