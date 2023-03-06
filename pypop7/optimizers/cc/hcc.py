import time

import numpy as np

from pypop7.optimizers.es.lmcma import LMCMA  # for upper-level
from pypop7.optimizers.es.cmaes import CMAES  # for lower-level
from pypop7.optimizers.cc import CC


class HCC(CC):
    """Hierarchical Cooperative Co-evolution (HCC).

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
       >>> from pypop7.optimizers.cc.hcc import HCC
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3*numpy.ones((2,))}
       >>> hcc = HCC(problem, options)  # initialize the optimizer class
       >>> results = hcc.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"HCC: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       HCC: 5001, 0.0026675385361584567

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

    Gomez, F.J. and Schmidhuber, J., 2005, June.
    Co-evolving recurrent neurons learn deep memory POMDPs.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 491-498). ACM.
    https://dl.acm.org/doi/10.1145/1068009.1068092
    """
    def __init__(self, problem, options):
        CC.__init__(self, problem, options)
        self.sigma = options.get('sigma')  # global step size
        assert self.sigma is None or self.sigma > 0.0
        self.ndim_subproblem = int(options.get('ndim_subproblem', 30))
        assert self.ndim_subproblem > 0

    def initialize(self, arg=None):
        # for lower-level CMA-ES
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
        # for upper-level LM-CMA
        problem = {'ndim_problem': self.ndim_problem,
                   'lower_boundary': self.lower_boundary,
                   'upper_boundary': self.upper_boundary}
        if self.sigma is None:
            sigma = np.min((self.upper_boundary - self.lower_boundary)/3.0)
        else:
            sigma = self.sigma
        options = {'seed_rng': self.rng_initialization.integers(np.iinfo(np.int64).max),
                   'sigma': sigma,
                   'max_runtime': self.max_runtime,
                   'verbose': False}
        lmcma = LMCMA(problem, options)
        return sub_optimizers, self.best_so_far_y, lmcma

    def optimize(self, fitness_function=None, args=None):
        fitness, is_initialization = CC.optimize(self, fitness_function), True
        used_fe = self.n_function_evaluations
        sub_optimizers, y, lmcma = self.initialize(args)
        # run for upper-level LM-CMA
        lm_mean, lm_x, lm_p_c, lm_s, lm_vm, lm_pm, lm_b, lm_d, lm_y = lmcma.initialize()  # for upper-level LM-CMA
        lmcma.start_time = time.time()
        lmcma.fitness_function = self.fitness_function
        # run for lower-level CMA-ES
        x_s, mean_s, ps_s, pc_s, cm_s, ee_s, ea_s, y_s = [], [], [], [], [], [], [], []  # for lower-level CMA-ES
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            y = []
            # run for upper-level LM-CMA
            lm_y_bak = np.copy(lm_y)
            lmcma.max_function_evaluations = self.max_function_evaluations - used_fe
            lm_x, lm_y = lmcma.iterate(lm_mean, lm_x, lm_pm, lm_vm, lm_y, lm_b, args)
            used_fe += lmcma.n_individuals
            self.n_function_evaluations += lmcma.n_individuals
            lm_mean, lm_p_c, lm_s, lm_vm, lm_pm, lm_b, lm_d = lmcma.update_distribution(
                lm_mean, lm_x, lm_p_c, lm_s, lm_vm, lm_pm, lm_b, lm_d, lm_y, lm_y_bak)
            y.extend(lm_y)
            if self.best_so_far_y > lmcma.best_so_far_y:
                self.best_so_far_x, self.best_so_far_y = lmcma.best_so_far_x, lmcma.best_so_far_y
            # run for lower-level CMA-ES
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
                for i, opt in enumerate(sub_optimizers):
                    ii = range(i*self.ndim_subproblem, np.minimum((i + 1)*self.ndim_subproblem, self.ndim_problem))
                    if self._check_terminations():
                        break

                    def sub_function(sub_x):  # to define sub-function for each sub-optimizer
                        best_so_far_x = np.copy(self.best_so_far_x)
                        best_so_far_x[ii] = sub_x
                        return self._evaluate_fitness(best_so_far_x, args)
                    opt.fitness_function = sub_function
                    opt.max_function_evaluations = self.max_function_evaluations - used_fe
                    x_s[i], y_s[i] = opt.iterate(x_s[i], mean_s[i], ee_s[i], ea_s[i], y_s[i], args)
                    opt.n_generations += 1
                    used_fe += opt.n_individuals
                    mean_s[i], ps_s[i], pc_s[i], cm_s[i], ee_s[i], ea_s[i] = opt.update_distribution(
                        x_s[i], mean_s[i], ps_s[i], pc_s[i], cm_s[i], ee_s[i], ea_s[i], y_s[i])
                    y.extend(y_s[i])
                if self.best_so_far_y < lmcma.best_so_far_y:
                    lm_mean = np.copy(self.best_so_far_x)
            self._n_generations += 1
        return self._collect(fitness, y)
