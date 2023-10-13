import numpy as np  # engine for numerical computing
from scipy.stats import norm

from pypop7.optimizers.es.es import ES


class MMES(ES):
    """Mixture Model-based Evolution Strategy (MMES).

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

                * 'm'             - number of candidate direction vectors (`int`, default:
                  `2*int(np.ceil(np.sqrt(problem['ndim_problem'])))`),
                * 'c_c'           - learning rate of evolution path update (`float`, default:
                  `0.4/np.sqrt(problem['ndim_problem'])`),
                * 'ms'            - mixing strength (`int`, default: `4`),
                * 'c_s'           - learning rate of global step-size adaptation (`float`, default: `0.3`),
                * 'a_z'           - target significance level (`float`, default: `0.05`),
                * 'distance'      - minimal distance of updating evolution paths (`int`, default:
                  `int(np.ceil(1.0/options['c_c']))`),
                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default:
                  `4 + int(3*np.log(problem['ndim_problem']))`),
                * 'n_parents'     - number of parents, aka parental population size (`int`, default:
                  `int(options['n_individuals']/2)`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.mmes import MMES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 200,
       ...            'lower_boundary': -5*numpy.ones((200,)),
       ...            'upper_boundary': 5*numpy.ones((200,))}
       >>> options = {'max_function_evaluations': 500000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((200,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> mmes = MMES(problem, options)  # initialize the optimizer class
       >>> results = mmes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"MMES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       MMES: 500000, 7.350414979801825

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/3ym72w5m>`_ for more details.

    Attributes
    ----------
    a_z           : `float`
                    target significance level.
    c_c           : `float`
                    learning rate of evolution path update.
    c_s           : `float`
                    learning rate of global step-size adaptation.
    distance      : `int`
                    minimal distance of updating evolution paths.
    m             : `int`
                    number of candidate direction vectors.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    ms            : `int`
                    mixing strength.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.
    sigma         : `float`
                    final global step-size, aka mutation strength.

    References
    ----------
    He, X., Zheng, Z. and Zhou, Y., 2021.
    MMES: Mixture model-based evolution strategy for large-scale optimization.
    IEEE Transactions on Evolutionary Computation, 25(2), pp.320-333.
    https://ieeexplore.ieee.org/abstract/document/9244595

    See the official Matlab version from He:
    https://github.com/hxyokokok/MMES
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        # set number of candidate direction vectors
        self.m = options.get('m', 2*int(np.ceil(np.sqrt(self.ndim_problem))))
        # set learning rate of evolution path
        self.c_c = options.get('c_c', 0.4/np.sqrt(self.ndim_problem))
        self.ms = options.get('ms', 4)  # mixing strength (l)
        # set for paired test adaptation (PTA)
        self.c_s = options.get('c_s', 0.3)  # learning rate of global step-size adaptation
        self.a_z = options.get('a_z', 0.05)  # target significance level
        # set minimal distance of updating evolution paths (T)
        self.distance = options.get('distance', int(np.ceil(1.0/self.c_c)))
        # set success probability of geometric distribution (different from 4/n in the original paper)
        self.c_a = options.get('c_a', 3.8/self.ndim_problem)  # same as Matlab code
        self.gamma = options.get('gamma', 1.0 - np.power(1.0 - self.c_a, self.m))
        self._n_mirror_sampling = None
        self._z_1 = np.sqrt(1.0 - self.gamma)
        self._z_2 = np.sqrt(self.gamma/self.ms)
        self._p_1 = 1.0 - self.c_c
        self._p_2 = np.sqrt(self.c_c*(2.0 - self.c_c))
        self._w_1 = 1.0 - self.c_s
        self._w_2 = np.sqrt(self.c_s*(2.0 - self.c_s))

    def initialize(self, args=None, is_restart=False):
        self._n_mirror_sampling = int(np.ceil(self.n_individuals/2))
        x = np.zeros((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        p = np.zeros((self.ndim_problem,))  # evolution path
        w = 0.0
        q = np.zeros((self.m, self.ndim_problem))  # candidate direction vectors
        t = np.zeros((self.m,))  # recorded generations
        v = np.arange(self.m)  # indexes to evolution paths
        y = np.tile(self._evaluate_fitness(mean, args), (self.n_individuals,))  # fitness
        return x, mean, p, w, q, t, v, y

    def iterate(self, x=None, mean=None, q=None, v=None, y=None, args=None):
        for k in range(self._n_mirror_sampling):  # mirror sampling
            zq = np.zeros((self.ndim_problem,))
            for _ in range(self.ms):
                j_k = v[(self.m - self.rng_optimization.geometric(self.c_a) % self.m) - 1]
                zq += self.rng_optimization.standard_normal()*q[j_k]
            z = self._z_1*self.rng_optimization.standard_normal((self.ndim_problem,))
            z += self._z_2*zq
            x[k] = mean + self.sigma*z
            if (self._n_mirror_sampling + k) < self.n_individuals:
                x[self._n_mirror_sampling + k] = mean - self.sigma*z
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _update_distribution(self, x=None, mean=None, p=None, w=None, q=None,
                             t=None, v=None, y=None, y_bak=None):
        order = np.argsort(y)[:self.n_parents]
        y.sort()
        mean_w = np.dot(self._w[:self.n_parents], x[order])
        p = self._p_1*p + self._p_2*np.sqrt(self._mu_eff)*(mean_w - mean)/self.sigma
        mean = mean_w
        if self._n_generations < self.m:
            q[self._n_generations] = p
        else:
            k_star = np.argmin(t[v[1:]] - t[v[:(self.m - 1)]])
            k_star += 1
            if t[v[k_star]] - t[v[k_star - 1]] > self.distance:
                k_star = 0
            v = np.append(np.append(v[:k_star], v[(k_star + 1):]), v[k_star])
            t[v[-1]], q[v[-1]] = self._n_generations, p
        # conduct success-based mutation strength adaptation
        l_w = np.dot(self._w, y_bak[:self.n_parents] > y[:self.n_parents])
        w = self._w_1*w + self._w_2*np.sqrt(self._mu_eff)*(2*l_w - 1)
        self.sigma *= np.exp(norm.cdf(w) - 1.0 + self.a_z)
        return mean, p, w, q, t, v

    def restart_reinitialize(self, args=None, x=None, mean=None, p=None, w=None, q=None,
                             t=None, v=None, y=None, fitness=None):
        if self.is_restart and ES.restart_reinitialize(self, y):
            x, mean, p, w, q, t, v, y = self.initialize(args, True)
            self._print_verbose_info(fitness, y[0])
        return x, mean, p, w, q, t, v, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p, w, q, t, v, y = self.initialize(args)
        self._print_verbose_info(fitness, y[0])
        while not self.termination_signal:
            y_bak = np.copy(y)
            # sample and evaluate offspring population
            x, y = self.iterate(x, mean, q, v, y, args)
            if self._check_terminations():
                break
            mean, p, w, q, t, v = self._update_distribution(x, mean, p, w, q, t, v, y, y_bak)
            self._n_generations += 1
            self._print_verbose_info(fitness, y)
            x, mean, p, w, q, t, v, y = self.restart_reinitialize(
                args, x, mean, p, w, q, t, v, y, fitness)
        results = self._collect(fitness, y, mean)
        results['p'] = p
        results['w'] = w
        return results
