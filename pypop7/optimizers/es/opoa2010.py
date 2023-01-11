import numpy as np

from pypop7.optimizers.es.opoc2009 import OPOC2009


class OPOA2010(OPOC2009):
    """(1+1)-Active-CMA-ES 2010 (OPOA2010).

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
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'sigma' - initial global step-size, aka mutation strength (`float`),
                * 'mean'  - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

    Examples
    --------
    Use the optimizer `OPOA2010` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.opoa2010 import OPOA2010
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> opoa2010 = OPOA2010(problem, options)  # initialize the optimizer class
       >>> results = opoa2010.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"OPOA2010: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       OPOA2010: 5000, 6.573983554197426e-16

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/26tad82p>`_ for more details.

    References
    ----------
    Arnold, D.V. and Hansen, N., 2010, July.
    Active covariance matrix adaptation for the (1+1)-CMA-ES.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 385-392). ACM.
    https://dl.acm.org/doi/abs/10.1145/1830483.1830556
    """
    def __init__(self, problem, options):
        OPOC2009.__init__(self, problem, options)
        self.c_m = options.get('c_m', 0.4/(np.power(self.ndim_problem, 1.6) + 1.0))
        self.k = options.get('k', 5)
        self._ancestors = []

    def initialize(self, args=None, is_restart=False):
        mean, y, a, a_i, best_so_far_y, p_s, p_c = OPOC2009.initialize(self, args, is_restart)
        return mean, y, a, a_i, best_so_far_y, p_s, p_c

    def iterate(self, mean=None, a=None, a_i=None, best_so_far_y=None, p_s=None, p_c=None, args=None):
        # sample and evaluate only one offspring
        z = self.rng_optimization.standard_normal((self.ndim_problem,))
        x = mean + self.sigma*np.dot(a, z)
        y = self._evaluate_fitness(x, args)
        l_s = 1 if y <= best_so_far_y else 0
        p_s = (1.0 - self.c_p)*p_s + self.c_p*l_s
        self.sigma *= np.exp(self.lr_sigma*(p_s - self.p_ts)/(1.0 - self.p_ts))
        if y <= best_so_far_y:
            self._ancestors.append(y)
            mean, best_so_far_y = x, y
            if p_s < self.p_t:
                p_c = (1.0 - self.c_c)*p_c + np.sqrt(self.c_c*(2.0 - self.c_c))*np.dot(a, z)
                alpha = 1.0 - self.c_cov
            else:
                p_c *= 1.0 - self.c_c
                alpha = 1.0 - self.c_cov + self.c_cov*self.c_c*(2.0 - self.c_c)
            w = np.dot(a_i, p_c)
            w_power = np.dot(w, w)
            alpha = np.sqrt(alpha)
            beta = alpha/w_power*(np.sqrt(1.0 + self.c_cov/(1.0 - self.c_cov)*w_power) - 1.0)
            a = alpha*a + beta*np.dot(p_c[:, np.newaxis], w[np.newaxis, :])
            a_i = 1.0/alpha*a_i - beta/(np.power(alpha, 2) + alpha*beta*w_power)*np.dot(
                w[:, np.newaxis], np.dot(w[np.newaxis, :], a_i))
        if len(self._ancestors) >= self.k and y > self._ancestors[-self.k]:
            del self._ancestors[0]
            z_power = np.dot(z, z)
            if 1.0 < self.c_m*(2.0*z_power - 1.0):
                c_m = 1.0/(2.0*z_power - 1.0)
            else:
                c_m = self.c_m
            alpha = np.sqrt(1.0 + c_m)
            beta = alpha/z_power*(np.sqrt(1.0 - self.c_m/(1.0 - self.c_m)*z_power) - 1.0)
            a = alpha*a + beta*np.dot(np.dot(a, z[:, np.newaxis]), z[np.newaxis, :])
            a_i = 1.0/alpha*a_i - beta/(np.power(alpha, 2) + alpha*beta*z_power)*np.dot(
                z[:, np.newaxis], np.dot(z[np.newaxis, :], a_i))
        return mean, y, a, a_i, best_so_far_y, p_s, p_c

    def restart_reinitialize(self, mean=None, y=None, a=None, a_i=None, best_so_far_y=None,
                             p_s=None, p_c=None, fitness=None, args=None):
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
            mean, y, a, a_i, best_so_far_y, p_s, p_c = self.initialize(args, True)
            self._list_fitness = [best_so_far_y]
            self._ancestors = []
        return mean, y, a, a_i, best_so_far_y, p_s, p_c
