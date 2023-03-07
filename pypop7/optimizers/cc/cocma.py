import time

import numpy as np

from pypop7.optimizers.es.cmaes import CMAES
from pypop7.optimizers.cc import CC


class COCMA(CC):
    """CoOperative CO-evolutionary Covariance Matrix Adaptation (COCMA).

    .. note:: For `COCMA`, `CMA-ES <https://pypop.readthedocs.io/en/latest/es/cmaes.html>`_ is used as the suboptimizer,
       since it could learn the variable dependencies in each subsapce to accelerate convergence. The simplest *cyclic*
       decomposition is employed to tackle **non-separable** objective functions, argurably a common feature of most
       real-world applications.

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
                * 'n_individuals'   - number of individuals/samples, aka population size (`int`, default: `100`).
                * 'sigma'           - initial global step-size (`float`, default:
                  `problem['upper_boundary'] - problem['lower_boundary']/3.0`),
                * 'ndim_subproblem' - dimensionality of subproblem for decomposition (`int`, default: `30`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
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
       COCMA: 5000, 5.610179991025547e-09

    For its correctness checking of coding, we cannot provide the code-based repeatability report, since this
    implementation combines different papers. To our knowledge, few well-designed open-source code of `CC` is
    available for non-separable black-box optimization.

    Attributes
    ----------
    n_individuals   : `int`
                      number of individuals/samples, aka population size.
    sigma           : `float`
                      initial global step-size.
    ndim_subproblem : `int`
                      dimensionality of subproblem for decomposition.

    References
    ----------
    Mei, Y., Omidvar, M.N., Li, X. and Yao, X., 2016.
    A competitive divide-and-conquer algorithm for unconstrained large-scale black-box optimization.
    ACM Transactions on Mathematical Software, 42(2), pp.1-24.
    https://dl.acm.org/doi/10.1145/2791291

    Potter, M.A. and De Jong, K.A., 1994, October.
    A cooperative coevolutionary approach to function optimization.
    In International Conference on Parallel Problem Solving from Nature (pp. 249-257).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/3-540-58484-6_269
    """
    def __init__(self, problem, options):
        CC.__init__(self, problem, options)
        self.sigma = options.get('sigma')  # global step size
        assert self.sigma is None or self.sigma > 0.0
        self.ndim_subproblem = int(options.get('ndim_subproblem', 30))
        assert self.ndim_subproblem > 0

    def initialize(self, arg=None):
        self.best_so_far_x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        self.best_so_far_y = self._evaluate_fitness(self.best_so_far_x, arg)
        sub_optimizers = []
        for i in range(int(np.ceil(self.ndim_problem/self.ndim_subproblem))):
            ii = range(i*self.ndim_subproblem, np.minimum((i + 1)*self.ndim_subproblem, self.ndim_problem))
            problem = {'ndim_problem': len(ii),  # cyclic decomposition
                       'lower_boundary': self.lower_boundary[ii],
                       'upper_boundary': self.upper_boundary[ii]}
            if self.sigma is None:
                sigma = np.min((self.upper_boundary[ii] - self.lower_boundary[ii])/3.0)
            else:
                sigma = self.sigma
            options = {'seed_rng': self.rng_initialization.integers(np.iinfo(np.int64).max),
                       'sigma': sigma,
                       'max_runtime': self.max_runtime,
                       'verbose': False}
            cma = CMAES(problem, options)
            cma.start_time = time.time()
            sub_optimizers.append(cma)
        return sub_optimizers, self.best_so_far_y

    def optimize(self, fitness_function=None, args=None):
        fitness, is_initialization = CC.optimize(self, fitness_function), True
        sub_optimizers, y = self.initialize(args)
        x_s, mean_s, ps_s, pc_s, cm_s, ee_s, ea_s, y_s = [], [], [], [], [], [], [], []
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
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
                y = []
                for i, opt in enumerate(sub_optimizers):
                    ii = range(i*self.ndim_subproblem, np.minimum((i + 1)*self.ndim_subproblem, self.ndim_problem))

                    def sub_function(sub_x):  # to define sub-function for each sub-optimizer
                        best_so_far_x = np.copy(self.best_so_far_x)
                        best_so_far_x[ii] = sub_x
                        return self._evaluate_fitness(best_so_far_x, args)
                    opt.fitness_function = sub_function
                    opt.max_function_evaluations = (opt.n_function_evaluations +
                                                    self.max_function_evaluations - self.n_function_evaluations)
                    x_s[i], y_s[i] = opt.iterate(x_s[i], mean_s[i], ee_s[i], ea_s[i], y_s[i], args)
                    y.extend(y_s[i])
                    if self._check_terminations():
                        break
                    opt.n_generations += 1
                    mean_s[i], ps_s[i], pc_s[i], cm_s[i], ee_s[i], ea_s[i] = opt.update_distribution(
                        x_s[i], mean_s[i], ps_s[i], pc_s[i], cm_s[i], ee_s[i], ea_s[i], y_s[i])
                self._n_generations += 1
        return self._collect(fitness, y)
