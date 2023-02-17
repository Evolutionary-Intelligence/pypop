import numpy as np
from scipy.stats import norm

from pypop7.optimizers.es.es import ES

class SDAES(ES):
    """search direction adaptation evolution strategy(SDA-ES)

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
              and with the following particular settings (`keys`):
                * 'sigma'             - initial global step-size, aka mutation strength (`float`),
                * 'mean'              - initial (starting) point, aka mean of Gaussian search distribution
                  (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'n_evolution_paths' - number of evolution paths (`int`, default: `10`),
                * 'n_individuals'     - number of offspring, aka offspring population size (`int`, default:
                  `4 + int(3*np.log(problem['ndim_problem']))`),
                * 'n_parents'         - number of parents, aka parental population size (`int`, default:
                  `int(options['n_individuals']/2)`),
                * 'c_cov'             - learning rate of low-rank covariance matrix (`float`, default:
                  `0.4 / np.sqrt(problem[ndim_problem]) `),
                * 'cc'                - learning rate for search direction adapation (`float`, default:
                  `0.25 / np.sqrt(problem[ndim_problem]) `),
                * 'd_s'               - damping factor  controlling the changing rate of the
                   mutation strength(`int`, default: `1 `),
                * 'c_s'               - learning rate of mutation strength (`float`, default: `0.3`),
                * 'q_target'          - target significant level (`float`, default: `0.05`).

    Examples
    --------
    Use the optimizer `SDAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.sdaes import SDAES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 200,
       ...            'lower_boundary': -5*numpy.ones((200,)),
       ...            'upper_boundary': 5*numpy.ones((200,))}
       >>> options = {'max_function_evaluations': 100000,  # set optimizer options
       ...            'seed_rng': 0,
       ...            'mean': 3*numpy.ones((200,)),
       ...            'sigma': 3.0,  # the global step-size may need to be tuned for better performance
       ...            'is_restart': False}
       >>> sdaes = SDAES(problem, options)  # initialize the optimizer class
       >>> results = sdaes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"SDAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       SDAES: 100000, 4.11321e-22

    Attributes
    ----------
    c_cov             : `float`
                        learning rate of low-rank covariance matrix
    cc                : `float`
                        learning rate for search direction adapation
    d_s               : `int`
                        damping factor  controlling the changing rate of the mutation strength
    c_s               : `float`
                        learning rate of mutation strength
    q_target          : `float`
                        target significant level
    mean              : `array_like`
                        initial (starting) point, aka mean of Gaussian search distribution.
    n_evolution_paths : `int`
                        number of evolution paths.
    n_individuals     : `int`
                        number of offspring, aka offspring population size.
    n_parents         : `int`
                        number of parents, aka parental population size.
    sigma             : `float`
                        final global step-size, aka mutation strength.

    References
    ----------
    He, X., Zhou, Y., Chen, Z., Zhang, J., & Chen, W. N. (2019).
    Large-scale evolution strategy based on search direction adaptation.
    IEEE Transactions on Cybernetics, 51(3), 1651-1665.
    https://ieeexplore.ieee.org/abstract/document/8781905

    See the official Matlab version from Xiaoyu He:
    https://github.com/hxyokokok/SDAES/tree/a3c7ac45f3b6e057d6effd45b0969dc464f40802
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.options = options
        self.n_evolution_paths = options.get('n_evolution_paths', 10)
        self.c_cov = 0.4 / np.sqrt(self.ndim_problem)
        self.cc = 0.25 / np.sqrt(self.ndim_problem)
        self.d_s = 1
        self.c_s = 0.3
        self.q_target = 0.05
        self._z_1 = np.sqrt(1 - self.c_cov)
        self._z_2 = np.sqrt(self.c_cov)
        self._s_1 = 1 - self.cc
        self._s_2 = np.sqrt(self.cc * (2 - self.cc))
        self._t_1 = 1 - self.c_s
        self._t_2 = np.sqrt(self.c_s * (2 - self.c_s))
        self._r_1 = np.power(self.n_individuals, 2) / 2
        self._r_2 = 1.0 / np.sqrt(np.power(self.n_individuals, 2) * (2 * self.n_individuals + 1) / 12)

    def initialize(self, is_restart=False):
        mean = self._initialize_mean(is_restart)
        zcum = 0
        q = np.empty((self.n_evolution_paths, self.ndim_problem))
        for i in range(self.n_evolution_paths):
            q[i] = 1e-6 * self.rng_optimization.standard_normal((self.ndim_problem,))
        x = np.empty((self.n_individuals, self.ndim_problem))
        y = self._evaluate_fitness(mean) * np.ones((self.n_individuals,))
        return x, mean, zcum, q, y

    def iterate(self, mean=None, q=None, x=None, y=None, args=None):
        for k in range(self.n_individuals):
            z_1 = self.rng_optimization.standard_normal((1, self.ndim_problem))
            z_2 = self.rng_optimization.standard_normal((1, self.n_evolution_paths))
            x[k] = mean + self.sigma * (self._z_1 * z_1 + self._z_2 * np.dot(z_2, q))
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _update_distributopn(self, mean=None, q=None, zcum=None, x=None, y=None, y_bak=None):
        order = np.argsort(y)
        new_mean = 0
        for i in range(self.n_parents):
            new_mean += self._w[i] * x[order[i]]
        z = np.sqrt(self._mu_eff) * (new_mean - mean) / self.sigma
        for i in range(self.n_evolution_paths):
            q[i] = self._s_1 * q[i] + self._s_2 * z
            z_t, q_t = z.reshape(1, self.ndim_problem), q[i].reshape(1, self.ndim_problem)
            t = np.matmul(z_t, q[i]) / np.matmul(q_t, q[i])
            z = (z - t * q[i]) / np.sqrt(1 + np.power(t, 2))
        new_order = np.argsort(np.concatenate((y_bak, y), axis=0))
        r1 = np.sum(new_order[:self.n_individuals])
        u = r1 - self.n_individuals * (self.n_individuals + 1) / 2
        zcum = self._t_1 * zcum + self._t_2 * (u - self._r_1) * self._r_2
        self.sigma *= np.exp((norm.cdf(zcum) / (1 - self.q_target) - 1) / self.d_s)
        return new_mean, q, zcum

    def restart_reinitialize(self, x=None, mean=None, zcum=None, q=None, y=None):
        if self.is_restart and ES.restart_reinitialize(self, y):
            x, mean, zcum, q, y = self.initialize(True)
        return x, mean, zcum, q, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, zcum, q, y = self.initialize(args)
        self._print_verbose_info(fitness, y[0])
        while not self._check_terminations():
            y_bak = np.copy(y)
            x, y = self.iterate(mean, q, x, y)
            self._print_verbose_info(fitness, y)
            mean, q, zcum = self._update_distributopn(mean, q, zcum, x, y, y_bak)
            self._n_generations += 1
        results = self._collect(fitness, y, mean)
        return results
