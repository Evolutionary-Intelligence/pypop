import time

import numpy as np

from pypop7.optimizers.es.cmaes import CMAES
from pypop7.optimizers.cc import CC


class COCMA(CC):
    """CoOperative CO-evolutionary Covariance Matrix Adaptation (COCMA).

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
    Use the optimizer `COCMA` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.cc.cocma import COCMA
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3*numpy.ones((2,))}
       >>> cocma = COCMA(problem, options)  # initialize the optimizer class
       >>> results = cocma.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"COCMA: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       COCMA: 5005, 5.610179991025547e-09

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
        self.ndim_subproblem = int(options.get('ndim_subproblem', 30))
        assert self.ndim_subproblem > 0, f'self.ndim_subproblem should > 0, but = {self.ndim_subproblem}.'

    def initialize(self, arg=None):
        self.best_so_far_x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        self.best_so_far_y = self._evaluate_fitness(self.best_so_far_x, arg)
        sub_optimizers = []
        for i in range(int(np.ceil(self.ndim_problem/self.ndim_subproblem))):
            ii = range(i*self.ndim_subproblem, np.minimum((i+1)*self.ndim_subproblem, self.ndim_problem))
            problem = {'ndim_problem': len(ii),  # cyclic decomposition
                       'lower_boundary': self.lower_boundary[ii],
                       'upper_boundary': self.upper_boundary[ii]}
            options = {'seed_rng': self.rng_initialization.integers(np.iinfo(np.int64).max),
                       'sigma': np.min((self.upper_boundary[ii] - self.lower_boundary[ii])/3.0),
                       'max_runtime': self.max_runtime,
                       'verbose': False}
            cma = CMAES(problem, options)
            cma.start_time = time.time()
            sub_optimizers.append(cma)
        return sub_optimizers, self.best_so_far_y

    def iterate(self):
        pass

    def optimize(self, fitness_function=None, args=None):
        fitness, is_initialization = CC.optimize(self, fitness_function), True
        sub_optimizers, y = self.initialize(args)
        x_s, mean_s, ps_s, pc_s, cm_s, ee_s, ea_s, y_s = [], [], [], [], [], [], [], []
        while not self._check_terminations():
            if is_initialization:
                is_initialization = False
                for i, opt in enumerate(sub_optimizers):
                    if self._check_terminations():
                        break

                    x, mean, p_s, p_c, cm, eig_ve, eig_va, yy = opt.initialize()
                    x_s.append(x)
                    mean_s.append(mean)
                    ps_s.append(p_s)
                    pc_s.append(p_c)
                    cm_s.append(cm)
                    ee_s.append(eig_ve)
                    ea_s.append(eig_va)
                    y_s.append(yy)
            else:
                self._print_verbose_info(fitness, y)
                y = []
                for i, opt in enumerate(sub_optimizers):
                    ii = range(i*self.ndim_subproblem, np.minimum((i + 1)*self.ndim_subproblem, self.ndim_problem))
                    if self._check_terminations():
                        break

                    def sub_function(sub_x):  # to define sub-function for each sub-optimizer
                        best_so_far_x = np.copy(self.best_so_far_x)
                        best_so_far_x[ii] = sub_x
                        return self._evaluate_fitness(best_so_far_x, args)
                    opt.fitness_function = sub_function
                    x_s[i], y_s[i] = opt.iterate(x_s[i], mean_s[i], ee_s[i], ea_s[i], y_s[i], args)
                    opt._n_generations += 1
                    mean_s[i], ps_s[i], pc_s[i], cm_s[i], ee_s[i], ea_s[i] = opt.update_distribution(
                        x_s[i], mean_s[i], ps_s[i], pc_s[i], cm_s[i], ee_s[i], ea_s[i], y_s[i])
                    y.extend(y_s[i])
            self._n_generations += 1
        return self._collect(fitness, y)
