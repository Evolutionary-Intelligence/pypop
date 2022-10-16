import numpy as np

from pypop7.optimizers.es.es import ES


class OPOC2006(ES):
    """(1+1)-Cholesky-CMA-ES 2006 (OPOC2006).

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

    Examples
    --------
    Use the `ES` optimizer `OPOC2006` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.opoc2006 import OPOC2006
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> opoc2006 = OPOC2006(problem, options)  # initialize the optimizer class
       >>> results = opoc2006.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"OPOC2006: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       OPOC2006: 5000, 4.487260671217195e-16

    References
    ----------
    Igel, C., Suttorp, T. and Hansen, N., 2006, July.
    A computational efficient covariance matrix update and a (1+1)-CMA for evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 453-460). ACM.
    https://dl.acm.org/doi/abs/10.1145/1143997.1144082
    (See Algorithm 2 for details.)
    """
    def __init__(self, problem, options):
        options['n_individuals'] = 1  # mandatory setting for OPOC2006
        options['n_parents'] = 1  # mandatory setting for OPOC2006
        ES.__init__(self, problem, options)
        if self.lr_sigma is None:
            self.lr_sigma = 1.0/(1.0 + self.ndim_problem/2.0)
        self.p_ts = options.get('p_ts', 2.0/11.0)
        self.c_p = options.get('c_p', 1.0/12.0)
        self.c_c = options.get('c_c', 2.0/(self.ndim_problem + 2.0))
        self.c_cov = options.get('c_cov', 2.0/(np.power(self.ndim_problem, 2) + 6.0))
        self.p_t = options.get('p_t', 0.44)
        self.c_a = options.get('c_a', np.sqrt(1.0 - self.c_cov))

    def initialize(self, args=None, is_restart=False):
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        y = self._evaluate_fitness(mean, args)  # fitness
        a = np.diag(np.ones(self.ndim_problem,))  # linear transformation (Cholesky factors)
        best_so_far_y, p_s = np.copy(y), self.p_ts
        return mean, y, a, best_so_far_y, p_s

    def iterate(self, args=None, mean=None, a=None, best_so_far_y=None, p_s=None):
        # sample and evaluate only one offspring
        z = self.rng_optimization.standard_normal((self.ndim_problem,))
        x = mean + self.sigma*np.dot(a, z)
        y = self._evaluate_fitness(x, args)
        if y <= best_so_far_y:
            l_s = 1
        else:
            l_s = 0
        p_s = (1.0 - self.c_p)*p_s + self.c_p*l_s
        self.sigma *= np.exp(self.lr_sigma*(p_s - self.p_ts)/(1.0 - self.p_ts))
        if y <= best_so_far_y:
            mean, best_so_far_y = x, y
            if p_s < self.p_t:
                z_norm, c_a = np.power(np.linalg.norm(z), 2), np.power(self.c_a, 2)
                a = self.c_a*a + self.c_a/z_norm*(np.sqrt(1.0 + ((1.0 - c_a)*z_norm)/c_a) - 1.0)*np.dot(
                    np.dot(a, z[:, np.newaxis]), z[np.newaxis, :])
        return mean, y, a, best_so_far_y, p_s

    def restart_reinitialize(self, args=None, mean=None, y=None, a=None,
                             best_so_far_y=None, p_s=None, fitness=None):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self._n_restart += 1
            self.sigma = np.copy(self._sigma_bak)
            mean, y, a, best_so_far_y, p_s = self.initialize(args, True)
            if self.saving_fitness:
                fitness.append(y)
            self._fitness_list = [best_so_far_y]
            self._n_generations = 0
            if self.verbose:
                print(' ....... restart .......')
            self._print_verbose_info(y)
        return mean, y, a, best_so_far_y, p_s

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        mean, y, a, best_so_far_y, p_s = self.initialize(args)
        if self.saving_fitness:
            fitness.append(y)
        self._print_verbose_info(y)
        while True:
            mean, y, a, best_so_far_y, p_s = self.iterate(args, mean, a, best_so_far_y, p_s)
            if self.saving_fitness:
                fitness.append(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                mean, y, a, best_so_far_y, p_s = self.restart_reinitialize(
                    args, mean, y, a, best_so_far_y, p_s, fitness)
        return self._collect_results(fitness, mean)
