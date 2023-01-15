import numpy as np

from pypop7.optimizers.es.es import ES


class R1ES(ES):
    """Rank-One Evolution Strategy (R1ES).

    .. note:: `R1ES` is a **low-rank** version of `CMA-ES` specifically designed for large-scale black-box optimization
       (LSBBO) by Li and `Zhang <https://tinyurl.com/32hsbx28>`_. It often works well when there is a *dominated* search
       direction embedded in a subspace. For more complex landscapes (e.g., there are multiple promising search
       directions), other LSBBO variants (e.g., `RMES`, `LMCMA`, `LMMAES`) of `CMA-ES` may be more preferred.

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
                * 'sigma'         - initial global step-size, aka mutation strength (`float`),
                * 'mean'          - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default:
                  `4 + int(3*np.log(problem['ndim_problem']))`),
                * 'n_parents'     - number of parents, aka parental population size (`int`, default:
                  `int(options['n_individuals']/2)`),
                * 'c_cov'         - learning rate of low-rank covariance matrix adaptation (`float`, default:
                  `1.0/(3.0*np.sqrt(problem['ndim_problem']) + 5.0)`),
                * 'c'             - learning rate of evolution path update (`float`, default:
                  `2.0/(problem['ndim_problem'] + 7.0)`),
                * 'c_s'           - learning rate of cumulative step-size adaptation (`float`, default: `0.3`),
                * 'q_star'        - baseline of cumulative step-size adaptation (`float`, default: `0.3`)，
                * 'd_sigma'       - delay factor of cumulative step-size adaptation (`float`, default: `1.0`).

    Examples
    --------
    Use the optimizer `R1ES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.r1es import R1ES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> r1es = R1ES(problem, options)  # initialize the optimizer class
       >>> results = r1es.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"R1ES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       R1ES: 5000, 8.942371004351231e-10

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/2aywpp2p>`_ for more details.

    Attributes
    ----------
    c               : `float`
                      learning rate of evolution path update.
    c_cov           : `float`
                      learning rate of low-rank covariance matrix adaptation.
    c_s             : `float`
                      learning rate of cumulative step-size adaptation.
    d_sigma         : `float`
                      delay factor of cumulative step-size adaptation.
    mean            : `array_like`
                      initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals   : `int`
                      number of offspring, aka offspring population size.
    n_parents       : `int`
                      number of parents, aka parental population size.
    q_star          : `float`
                      baseline of cumulative step-size adaptation.
    sigma           : `float`
                      final global step-size, aka mutation strength.

    References
    ----------
    Li, Z. and Zhang, Q., 2018.
    A simple yet efficient evolution strategy for large-scale black-box optimization.
    IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
    https://ieeexplore.ieee.org/abstract/document/8080257
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.c_cov = options.get('c_cov', 1.0/(3.0*np.sqrt(self.ndim_problem) + 5.0))
        self.c = options.get('c', 2.0/(self.ndim_problem + 7.0))
        self.c_s = options.get('c_s', 0.3)
        self.q_star = options.get('q_star', 0.3)
        self.d_sigma = options.get('d_sigma', 1.0)
        self._x_1 = np.sqrt(1.0 - self.c_cov)
        self._x_2 = np.sqrt(self.c_cov)
        self._p_1 = 1.0 - self.c
        self._p_2 = None
        self._rr = None  # for rank-based success rule (RSR)

    def initialize(self, args=None, is_restart=False):
        self._p_2 = np.sqrt(self.c*(2.0 - self.c)*self._mu_eff)
        self._rr = np.arange(self.n_parents*2) + 1  # for rank-based success rule (RSR)
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        p = np.zeros((self.ndim_problem,))  # principal search direction
        s = 0.0  # cumulative rank rate
        y = np.tile(self._evaluate_fitness(mean, args), (self.n_individuals,))  # fitness
        return x, mean, p, s, y

    def iterate(self, x=None, mean=None, p=None, y=None, args=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            z = self.rng_optimization.standard_normal((self.ndim_problem,))
            r = self.rng_optimization.standard_normal()
            x[k] = mean + self.sigma*(self._x_1*z + self._x_2*r*p)
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _update_distribution(self, x=None, mean=None, p=None, s=None, y=None, y_bak=None):
        order = np.argsort(y)[:self.n_parents]
        y.sort()
        mean_w = np.dot(self._w[:self.n_parents], x[order])
        p = self._p_1*p + self._p_2*(mean_w - mean)/self.sigma
        mean = mean_w
        r = np.argsort(np.hstack((y_bak[:self.n_parents], y[:self.n_parents])))
        rr = self._rr[r < self.n_parents] - self._rr[r >= self.n_parents]
        q = np.dot(self._w, rr)/self.n_parents
        s = (1.0 - self.c_s)*s + self.c_s*(q - self.q_star)
        self.sigma *= np.exp(s/self.d_sigma)
        return mean, p, s

    def restart_reinitialize(self, args=None, x=None, mean=None, p=None, s=None, y=None, fitness=None):
        if self.is_restart and ES.restart_reinitialize(self, y):
            x, mean, p, s, y = self.initialize(args, True)
            self._print_verbose_info(fitness, y[0])
            self.d_sigma *= 2.0
        return x, mean, p, s, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p, s, y = self.initialize(args)
        self._print_verbose_info(fitness, y[0])
        while not self._check_terminations():
            y_bak = np.copy(y)
            # sample and evaluate offspring population
            x, y = self.iterate(x, mean, p, y, args)
            self._n_generations += 1
            self._print_verbose_info(fitness, y)
            mean, p, s = self._update_distribution(x, mean, p, s, y, y_bak)
            x, mean, p, s, y = self.restart_reinitialize(args, x, mean, p, s, y, fitness)
        results = self._collect(fitness, y, mean)
        results['p'] = p
        results['s'] = s
        return results
