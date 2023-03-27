import numpy as np

from pypop7.optimizers.es.es import ES


class CCMAES2009(ES):
    """Cholesky-CMA-ES 2009 (CCMAES2009).

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
                * 'sigma'         - initial global step-size, aka mutation strength (`float`),
                * 'mean'          - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default:
                  `4 + int(3*np.log(problem['ndim_problem']))`),
                * 'n_parents'     - number of parents, aka parental population size (`int`, default:
                  `int(options['n_individuals']/2)`).

    Examples
    --------
    Use the optimizer `CCMAES2009` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.ccmaes2009 import CCMAES2009
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> ccmaes2009 = CCMAES2009(problem, options)  # initialize the optimizer class
       >>> results = ccmaes2009.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CCMAES2009: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CCMAES2009: 5000, 5.74495131488279e-17

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/c5hreha9>`_ for more details.

    References
    ----------
    Suttorp, T., Hansen, N. and Igel, C., 2009.
    Efficient covariance matrix update for variable metric evolution strategies.
    Machine Learning, 75(2), pp.167-197.
    https://link.springer.com/article/10.1007/s10994-009-5102-1
    (See Algorithm 4 for details.)
    """
    def __init__(self, problem, options):
        self.options = options
        ES.__init__(self, problem, options)
        self.c_s = None
        self.d_s = None
        self.c_c = options.get('c_c', 4.0/(self.ndim_problem + 4.0))
        self.c_cov = options.get('c_cov', 2.0/np.power(self.ndim_problem + np.sqrt(2.0), 2.0))

    def _set_c_s(self):
        return np.sqrt(self._mu_eff)/(np.sqrt(self.ndim_problem) + np.sqrt(self._mu_eff))

    def _set_d_s(self):
        return 1.0 + 2.0*np.maximum(0.0, np.sqrt((self._mu_eff - 1.0)/(self.ndim_problem + 1.0)) - 1.0) + self.c_s

    def initialize(self, is_restart=False):
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        z = np.empty((self.n_individuals, self.ndim_problem))  # Gaussian noise for mutation
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        a = np.diag(np.ones(self.ndim_problem,))  # Cholesky factors
        a_i = np.diag(np.ones(self.ndim_problem,))  # inverse of Cholesky factors
        p_s = np.zeros((self.ndim_problem,))  # evolution path for global step-size adaptation
        p_c = np.zeros((self.ndim_problem,))  # evolution path for covariance matrix adaptation
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        self.c_s = self.options.get('c_s', self._set_c_s())
        self.d_s = self.options.get('d_s', self._set_d_s())
        return mean, z, x, a, a_i, p_s, p_c, y

    def iterate(self, z=None, x=None, mean=None, a=None, y=None, args=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return z, x, y
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            x[k] = mean + self.sigma*np.dot(a, z[k])
            y[k] = self._evaluate_fitness(x[k], args)
        return z, x, y

    def _update_distribution(self, z=None, x=None, a=None, a_i=None, p_s=None, p_c=None, y=None):
        order = np.argsort(y)[:self.n_parents]
        mean, z_w = np.dot(self._w, x[order]), np.dot(self._w, z[order])
        p_c = (1.0-self.c_c)*p_c + np.sqrt(self.c_c*(2.0 - self.c_c)*self._mu_eff)*np.dot(a, z_w)
        v = np.dot(a_i, p_c)
        v_norm = np.dot(v, v)  # (||v||)^2
        s_v_norm = np.sqrt(1.0 + self.c_cov/(1.0 - self.c_cov)*v_norm)
        a_i = (a_i - (1.0 - 1.0/s_v_norm)*np.dot(v[:, np.newaxis], np.dot(v[np.newaxis, :], a_i))/v_norm
               )/np.sqrt(1.0 - self.c_cov)
        a = np.sqrt(1.0 - self.c_cov)*(a + (s_v_norm - 1.0)*np.outer(p_c, v)/v_norm)
        p_s = (1.0 - self.c_s)*p_s + np.sqrt(self.c_s*(2.0 - self.c_s)*self._mu_eff)*z_w
        self.sigma *= np.exp(self.c_s/self.d_s*(np.linalg.norm(p_s)/self._e_chi - 1.0))
        return mean, a, a_i, p_s, p_c

    def restart_reinitialize(self, mean=None, z=None, x=None, a=None, a_i=None, p_s=None, p_c=None, y=None):
        if self.is_restart and ES.restart_reinitialize(self, y):
            mean, z, x, a, a_i, p_s, p_c, y = self.initialize(True)
        return mean, z, x, a, a_i, p_s, p_c, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        mean, z, x, a, a_i, p_s, p_c, y = self.initialize()
        while not self.termination_signal:
            # sample and evaluate offspring population
            z, x, y = self.iterate(z, x, mean, a, y, args)
            if self._check_terminations():
                break
            mean, a, a_i, p_s, p_c = self._update_distribution(z, x, a, a_i, p_s, p_c, y)
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            mean, z, x, a, a_i, p_s, p_c, y = self.restart_reinitialize(
                mean, z, x, a, a_i, p_s, p_c, y)
        return self._collect(fitness, y, mean)
