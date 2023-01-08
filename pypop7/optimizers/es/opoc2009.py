import numpy as np

from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.opoc2006 import OPOC2006


class OPOC2009(OPOC2006):
    """(1+1)-Cholesky-CMA-ES 2009 (OPOC2009).

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
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`).
              and with the following particular settings (`keys`):
                * 'sigma' - initial global step-size, aka mutation strength (`float`),
                * 'mean'  - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

    Examples
    --------
    Use the optimizer `OPOC2009` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.opoc2009 import OPOC2009
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> opoc2009 = OPOC2009(problem, options)  # initialize the optimizer class
       >>> results = opoc2009.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"OPOC2009: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       OPOC2009: 5000, 2.7686802211556655e-17

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/3ba4ctny>`_ for more details.

    References
    ----------
    Suttorp, T., Hansen, N. and Igel, C., 2009.
    Efficient covariance matrix update for variable metric evolution strategies.
    Machine Learning, 75(2), pp.167-197.
    https://link.springer.com/article/10.1007/s10994-009-5102-1
    (See Algorithm 2 for details.)
    """
    def __init__(self, problem, options):
        OPOC2006.__init__(self, problem, options)

    def initialize(self, args=None, is_restart=False):
        mean, y, a, best_so_far_y, p_s = OPOC2006.initialize(self, args, is_restart)
        p_c = np.zeros((self.ndim_problem,))  # evolution path
        a_i = np.diag(np.ones((self.ndim_problem,)))  # inverse of Cholesky factors
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
            mean, best_so_far_y = x, y
            if p_s < self.p_t:
                p_c = (1.0 - self.c_c)*p_c + np.sqrt(self.c_c*(2.0 - self.c_c))*np.dot(a, z)
                alpha = 1.0 - self.c_cov
            else:
                p_c *= 1.0 - self.c_c
                alpha = 1.0 - self.c_cov + self.c_cov*self.c_c*(2.0 - self.c_c)
            beta = self.c_cov
            w = np.dot(a_i, p_c)
            w_norm = np.power(np.linalg.norm(w), 2)
            s_w_norm = np.sqrt(1.0 + beta/alpha*w_norm)
            a = np.sqrt(alpha)*a + np.sqrt(alpha)/w_norm*(s_w_norm - 1.0)*np.dot(
                p_c[:, np.newaxis], w[np.newaxis, :])
            a_i = 1.0/np.sqrt(alpha)*a_i - 1.0/(np.sqrt(alpha)*w_norm)*(1.0 - 1.0/s_w_norm)*np.dot(
                w[:, np.newaxis], np.dot(w[np.newaxis, :], a_i))
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
        return mean, y, a, a_i, best_so_far_y, p_s, p_c

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        mean, y, a, a_i, best_so_far_y, p_s, p_c = self.initialize(args)
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            mean, y, a, a_i, best_so_far_y, p_s, p_c = self.iterate(
                mean, a, a_i, best_so_far_y, p_s, p_c, args)
            self._n_generations += 1
            if self.is_restart:
                mean, y, a, a_i, best_so_far_y, p_s, p_c = self.restart_reinitialize(
                    mean, y, a, a_i, best_so_far_y, p_s, p_c, fitness, args)
        return self._collect(fitness, y, mean)
