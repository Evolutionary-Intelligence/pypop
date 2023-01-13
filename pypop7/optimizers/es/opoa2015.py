import numpy as np
import numba as nb

from pypop7.optimizers.es.es import ES


@nb.jit(nopython=True)
def cholesky_update(rm, z, downdate):
    # https://github.com/scipy/scipy/blob/d20f92fce9f1956bfa65365feeec39621a071932/
    #     scipy/linalg/_decomp_cholesky_update.py
    rm, z, alpha, beta = rm.T, z, np.empty_like(z), np.empty_like(z)
    alpha[-1], beta[-1] = 1.0, 1.0
    sign = -1 if downdate else 1
    for r in range(len(z)):
        a = z[r]/rm[r, r]
        alpha[r] = alpha[r - 1] + sign*np.power(a, 2)
        beta[r] = np.sqrt(alpha[r])
        z[r + 1:] -= a*rm[r, r + 1:]
        rm[r, r:] *= beta[r]/beta[r - 1]
        rm[r, r + 1:] += sign*a/(beta[r]*beta[r - 1])*z[r + 1:]
    return rm.T


class OPOA2015(ES):
    """(1+1)-Active-CMA-ES 2015 (OPOA2015).

    Parameters
    ----------
    problem : `dict`
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : `dict`
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`),
              and with the following particular settings (`keys`):
                * 'sigma' - initial global step-size, aka mutation strength (`float`),
                * 'mean'  - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

    Examples
    --------
    Use the optimizer `OPOA2015` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.opoa2015 import OPOA2015
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> opoa2015 = OPOA2015(problem, options)  # initialize the optimizer class
       >>> results = opoa2015.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"OPOA2015: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       OPOA2015: 5000, 5.955151843487958e-17

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/mrxu4suj>`_ for more details.

    References
    ----------
    Krause, O. and Igel, C., 2015, January.
    A more efficient rank-one covariance matrix update for evolution strategies.
    In Proceedings of ACM Conference on Foundations of Genetic Algorithms (pp. 129-136).
    https://dl.acm.org/doi/abs/10.1145/2725494.2725496
    """
    def __init__(self, problem, options):
        options['n_individuals'] = 1  # mandatory setting
        options['n_parents'] = 1  # mandatory setting
        ES.__init__(self, problem, options)
        if self.lr_sigma is None:
            self.lr_sigma = 1.0/(1.0 + self.ndim_problem/2.0)
        self.p_ts = options.get('p_ts', 2.0/11.0)
        self.c_p = options.get('c_p', 1.0/12.0)
        self.c_c = options.get('c_c', 2.0/(self.ndim_problem + 2.0))
        self.c_cov = options.get('c_cov', 2.0/(np.power(self.ndim_problem, 2) + 6.0))
        self.p_t = options.get('p_t', 0.44)
        self.c_m = options.get('c_m', 0.4/(np.power(self.ndim_problem, 1.6) + 1.0))
        self.k = options.get('k', 5)
        self._ancestors = []
        self._c_cf = 1.0 - self.c_cov + self.c_cov*self.c_c*(2.0 - self.c_c)

    def initialize(self, args=None, is_restart=False):
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        y = self._evaluate_fitness(mean, args)  # fitness
        cf = np.diag(np.ones(self.ndim_problem,))  # Cholesky factorization
        best_so_far_y, p_s = np.copy(y), self.p_ts
        p_c = np.zeros((self.ndim_problem,))  # evolution path
        return mean, y, cf, best_so_far_y, p_s, p_c

    def _cholesky_update(self, cf=None, alpha=None, beta=None, v=None):  # triangular rank-one update
        assert self.ndim_problem == v.size
        if beta < 0:
            downdate, beta = True, -beta
        else:
            downdate = False
        return cholesky_update(np.sqrt(alpha)*cf, np.sqrt(beta)*v, downdate)

    def iterate(self, mean=None, cf=None, best_so_far_y=None, p_s=None, p_c=None, args=None):
        # sample and evaluate only one offspring
        z = self.rng_optimization.standard_normal((self.ndim_problem,))
        cf_z = np.dot(cf, z)
        x = mean + self.sigma*cf_z
        y = self._evaluate_fitness(x, args)
        if y <= best_so_far_y:
            self._ancestors.append(y)
            mean, best_so_far_y = x, y
            p_s = (1.0 - self.c_p)*p_s + self.c_p
            is_better = True
        else:
            p_s *= 1.0 - self.c_p
            is_better = False
        self.sigma *= np.exp(self.lr_sigma*(p_s - self.p_ts)/(1.0 - self.p_ts))
        if p_s >= self.p_t:
            p_c *= 1.0 - self.c_c
            cf = self._cholesky_update(cf, self._c_cf, self.c_cov, p_c)
        elif is_better:
            p_c = (1.0 - self.c_c)*p_c + np.sqrt(self.c_c*(2.0 - self.c_c))*cf_z
            cf = self._cholesky_update(cf, 1.0 - self.c_cov, self.c_cov, p_c)
        elif len(self._ancestors) >= self.k and y > self._ancestors[-self.k]:
            del self._ancestors[0]
            c_m = np.minimum(self.c_m, 1.0/(2.0*np.dot(z, z) - 1.0))
            cf = self._cholesky_update(cf, 1.0 + c_m, -c_m, cf_z)
        self._n_generations += 1
        return mean, y, cf, best_so_far_y, p_s, p_c

    def restart_reinitialize(self, args=None, mean=None, y=None, cf=None, best_so_far_y=None,
                             p_s=None, p_c=None, fitness=None):
        self._list_fitness.append(best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._list_fitness) >= self.stagnation:
            is_restart_2 = (self._list_fitness[-self.stagnation] - self._list_fitness[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self._print_verbose_info(fitness, y, True)
            if self.verbose:
                print(' ....... *** restart *** .......')
            self._n_restart += 1
            self._list_generations.append(self._n_generations)  # for each restart
            self._n_generations = 0
            self.sigma = np.copy(self._sigma_bak)
            mean, y, cf, best_so_far_y, p_s, p_c = self.initialize(args, True)
            self._list_fitness = [best_so_far_y]
            self._ancestors = []
        return mean, y, cf, best_so_far_y, p_s, p_c

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        mean, y, cf, best_so_far_y, p_s, p_c = self.initialize(args)
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            mean, y, cf, best_so_far_y, p_s, p_c = self.iterate(
                mean, cf, best_so_far_y, p_s, p_c, args)
            if self.is_restart:
                mean, y, cf, best_so_far_y, p_s, p_c = self.restart_reinitialize(
                    args, mean, y, cf, best_so_far_y, p_s, p_c, fitness)
        return self._collect(fitness, y, mean)
