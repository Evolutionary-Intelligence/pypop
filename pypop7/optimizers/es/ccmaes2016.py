import numpy as np
from scipy.linalg import solve_triangular

from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.opoa2015 import cholesky_update


class CCMAES2016(ES):
    """Cholesky-CMA-ES 2016 (CCMAES2016).

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
    Use the optimizer `CCMAES2016` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.ccmaes2016 import CCMAES2016
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> ccmaes2016 = CCMAES2016(problem, options)  # initialize the optimizer class
       >>> results = ccmaes2016.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CCMAES2016: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CCMAES2016: 5000, 2.614231350522262e-21

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/3kbet9tn>`_ for more details.

    References
    ----------
    Krause, O., Arbonès, D.R. and Igel, C., 2016.
    CMA-ES with optimal covariance update and storage complexity.
    Advances in Neural Information Processing Systems, 29, pp.370-378.
    https://proceedings.neurips.cc/paper/2016/hash/289dff07669d7a23de0ef88d2f7129e7-Abstract.html
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.options = options
        self.c_s, self.d = None, None
        self.c_c, self.c_1, self.c_mu = None, None, None

    def initialize(self, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        a = np.diag(np.ones((self.ndim_problem,)))  # cholesky factor
        p_s = np.zeros((self.ndim_problem,))  # evolution path for CSA
        p_c = np.zeros((self.ndim_problem,))  # evolution path for CMA
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        self.c_s = self.options.get('c_s', self._mu_eff/(self.ndim_problem + self._mu_eff))
        self.d = self.options.get('d', 1.0 + np.sqrt(self._mu_eff/self.ndim_problem))
        self.c_c = self.options.get('c_c', (4.0 + self._mu_eff/self.ndim_problem)/(
                self.ndim_problem + 4.0 + 2.0*self._mu_eff/self.ndim_problem))
        self.c_1 = self.options.get('c_1', 2.0/(np.square(self.ndim_problem) + self._mu_eff))
        self.c_mu = self.options.get('c_mu', self._mu_eff/(np.square(self.ndim_problem) + self._mu_eff))
        return x, mean, a, p_s, p_c, y

    def iterate(self, x=None, mean=None, a=None, y=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            x[k] = mean + self.sigma*np.dot(a, self.rng_optimization.standard_normal((self.ndim_problem,)))
            y[k] = self._evaluate_fitness(x[k])
        return x, y

    def _update_distribution(self, x=None, mean=None, a=None, p_s=None, p_c=None, y=None):
        order = np.argsort(y)[:self.n_parents]
        mean_bak = np.dot(self._w, x[order])
        mean_diff = (mean_bak - mean)/self.sigma
        p_c = (1.0 - self.c_c)*p_c + np.sqrt(self.c_c*(2.0 - self.c_c)*self._mu_eff)*mean_diff
        p_s = (1.0 - self.c_s)*p_s + np.sqrt(self.c_s*(2.0 - self.c_s)*self._mu_eff)*solve_triangular(
            a, mean_diff, lower=True)
        a *= np.sqrt(1.0 - self.c_1 - self.c_mu)
        a = cholesky_update(a, np.sqrt(self.c_1)*p_c, False)
        for i in range(self.n_parents):
            a = cholesky_update(a, np.sqrt(self.c_mu*self._w[i])*(x[order[i]] - mean)/self.sigma, False)
        self.sigma *= np.exp(self.c_s/self.d*(np.sqrt(np.dot(p_s, p_s))/self._e_chi - 1.0))
        return mean_bak, a, p_s, p_c

    def restart_reinitialize(self, x=None, mean=None, a=None, p_s=None, p_c=None, y=None):
        if self.is_restart and ES.restart_reinitialize(self, y):
            x, mean, a, p_s, p_c, y = self.initialize(True)
        return x, mean, a, p_s, p_c, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, a, p_s, p_c, y = self.initialize()
        while not self._check_terminations():
            x, y = self.iterate(x, mean, a, y)  # sample and evaluate offspring population
            mean, a, p_s, p_c = self._update_distribution(x, mean, a, p_s, p_c, y)
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            x, mean, a, p_s, p_c, y = self.restart_reinitialize(x, mean, a, p_s, p_c, y)
        results = self._collect(fitness, y, mean)
        results['p_s'] = p_s
        results['p_c'] = p_c
        return results
