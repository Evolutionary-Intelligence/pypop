import numpy as np

from pypop7.optimizers.es.es import ES


class LMCMA(ES):
    """Limited-Memory Covariance Matrix Adaptation (LMCMA).

    .. note:: Currently `LMCMA` is a **State-Of-The-Art** variant of `CMA-ES` designed especially for large-scale
       black-box optimization. Inspired by `L-BFGS` (a well-established *second-order* gradient-based optimizer),
       it stores only *m* direction vectors to reconstruct the covariance matirx on-the-fly, resulting in **O(mn)**
       time complexity w.r.t. each sampling, where *m=O(log(n))* and *n* is the dimensionality of objective function.

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

                * 'm'             - number of direction vectors (`int`, default:
                  `4 + int(3*np.log(self.ndim_problem))`),
                * 'base_m'        - base number of direction vectors (`int`, default: `4`),
                * 'period'        - update period (`int`, default: `int(np.maximum(1, np.log(self.ndim_problem)))`),
                * 'n_steps'       - target number of generations between vectors (`int`, default: `self.ndim_problem`),
                * 'c_c'           - learning rate for evolution path update (`float`, default:
                  `0.5/np.sqrt(self.ndim_problem)`),
                * 'c_1'           - learning rate for covariance matrix adaptation (`float`, default:
                  `1.0/(10.0*np.log(self.ndim_problem + 1.0))`),
                * 'c_s'           - learning rate for population success rule (`float`, default: `0.3`),
                * 'd_s'           - changing rate for population success rule (`float`, default: `1.0`),
                * 'z_star'        - target success rate for population success rule (`float`, default: `0.3`),
                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default:
                  `4 + int(3*np.log(self.ndim_problem))`),
                * 'n_parents'     - number of parents, aka parental population size (`int`, default:
                  `int(self.n_individuals/2)`).

    Examples
    --------
    Use the `ES` optimizer `LMCMA` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.lmcma import LMCMA
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> lmcma = LMCMA(problem, options)  # initialize the optimizer class
       >>> results = lmcma.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"LMCMA: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       LMCMA: 5000, 7.54648196471487e-07

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/24jnfhbs>`_ for more details.

    Attributes
    ----------
    base_m        : `int`
                    base number of direction vectors.
    c_c           : `float`
                    learning rate for evolution path update.
    c_s           : `float`
                    learning rate for population success rule.
    c_1           : `float`
                    learning rate for covariance matrix adaptation.
    d_s           : `float`
                    changing rate for population success rule.
    m             : `int`
                    number of direction vectors.
    mean          : `array_like`
                    mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring (λ: lambda), offspring population size.
    n_parents     : `int`
                    number of parents (μ: mu), parental population size.
    n_steps       : `int`
                    target number of generations between vectors.
    period        : `int`
                    update period.
    sigma         : `float`
                    mutation strength.
    z_star        : `float`
                    target success rate for population success rule.

    References
    ----------
    Loshchilov, I., 2017.
    LM-CMA: An alternative to L-BFGS for large-scale black box optimization.
    Evolutionary Computation, 25(1), pp.143-171.
    https://direct.mit.edu/evco/article-abstract/25/1/143/1041/LM-CMA-An-Alternative-to-L-BFGS-for-Large-Scale
    (See Algorithm 7 for details.)

    See the official C++ version from Loshchilov, which provides an interface for Matlab users:
    https://sites.google.com/site/ecjlmcma/
    (Unfortunately, this website link appears to be not available now.)
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.m = options.get('m', 4 + int(3*np.log(self.ndim_problem)))  # number of direction vectors
        assert 0 < self.m <= self.ndim_problem
        self.base_m = options.get('base_m', 4)  # base number of direction vectors
        self.period = options.get('period', int(np.maximum(1, np.log(self.ndim_problem))))  # update period
        self.n_steps = options.get('n_steps', self.ndim_problem)  # target number of generations between vectors
        self.c_c = options.get('c_c', 0.5/np.sqrt(self.ndim_problem))  # learning rate for evolution path
        # set learning rate for covariance matrix adaptation (CMA)
        self.c_1 = options.get('c_1', 1.0/(10.0*np.log(self.ndim_problem + 1.0)))
        self.c_s = options.get('c_s', 0.3)  # learning rate for population success rule (PSR)
        self.d_s = options.get('d_s', 1.0)  # changing rate for PSR
        self.z_star = options.get('z_star', 0.3)  # target success rate for PSR
        self._a = np.sqrt(1.0 - self.c_1)
        self._c = 1.0/np.sqrt(1.0 - self.c_1)
        self._bd_1 = np.sqrt(1.0 - self.c_1)
        self._bd_2 = self.c_1/(1.0 - self.c_1)
        self._p_c_1 = 1.0 - self.c_c
        self._p_c_2 = None
        self._j = None
        self._l = None
        self._it = None
        self._rr = None  # for PSR

    def initialize(self, is_restart=False):
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        p_c = np.zeros((self.ndim_problem,))  # evolution path
        s = 0.0  # for PSR
        vm = np.empty((self.m, self.ndim_problem))
        pm = np.empty((self.m, self.ndim_problem))
        b = np.empty((self.m,))
        d = np.empty((self.m,))
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        self._p_c_2 = np.sqrt(self.c_c*(2.0 - self.c_c)*self._mu_eff)
        self._j = [None]*self.m
        self._l = [None]*self.m
        self._it = 0
        self._rr = np.arange(self.n_individuals*2, 0, -1) - 1
        return mean, x, p_c, s, vm, pm, b, d, y

    def _rademacher(self):
        """Sampling from Rademacher distribution."""
        random = self.rng_optimization.integers(2, size=(self.ndim_problem,))
        random[random == 0] = -1
        return np.double(random)

    def _a_z(self, z=None, pm=None, vm=None, b=None, start=None, it=None):
        """Algorithm 3 Az(): Cholesky factor-vector update."""
        x = np.copy(z)
        for t in range(start, it):
            x = self._a*x + b[self._j[t]]*np.dot(vm[self._j[t]], z)*pm[self._j[t]]
        return x

    def iterate(self, mean=None, x=None, pm=None, vm=None, y=None, b=None, args=None):
        sign, a_z = 1, np.empty((self.ndim_problem,))  # for mirrored sampling
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            if sign == 1:  # Algorithm 6 SelectSubst(): direction vectors selection
                base_m = (10.0*self.base_m if k == 0 else self.base_m)*np.abs(
                    self.rng_optimization.standard_normal())
                base_m = float(self._it if base_m > self._it else base_m)
                a_z = self._a_z(self._rademacher(), pm, vm, b,
                                int(self._it - base_m) if self._it > 1 else 0, self._it)
            x[k] = mean + sign*self.sigma*a_z
            y[k] = self._evaluate_fitness(x[k], args)
            sign *= -1  # sampling in the opposite direction for mirrored sampling
        return x, y

    def _a_inv_z(self, v=None, vm=None, d=None, i=None):
        """Algorithm 4 Ainvz(): inverse Cholesky factor-vector update."""
        x = np.copy(v)
        for t in range(0, i):
            x = self._c*x - d[self._j[t]]*np.dot(vm[self._j[t]], x)*vm[self._j[t]]
        return x

    def _update_distribution(self, mean=None, x=None, p_c=None, s=None, vm=None, pm=None,
                             b=None, d=None, y=None, y_bak=None):
        mean_bak = np.dot(self._w, x[np.argsort(y)[:self.n_parents]])
        p_c = self._p_c_1*p_c + self._p_c_2*(mean_bak - mean)/self.sigma
        # select and store direction vectors - to preserve a certain temporal distance in terms of
        #   number of generations between the stored direction vectors (Algorithm 5)
        if self._n_generations % self.period == 0:  # temporal distance
            _n_generations = int(self._n_generations/self.period)  # temporal distance
            i_min = 1  # index of the first vector that will be replaced by the new one
            # the higher the index of `self._j`, the more recent is the corresponding direction vector
            if _n_generations < self.m:
                self._j[_n_generations] = _n_generations
            else:
                if self.m > 1:
                    # find a pair of consecutively saved vectors with the distance between them
                    #   closest to a target distance
                    d_min = (self._l[self._j[1]] - self._l[self._j[0]]) - self.n_steps
                    for j in range(2, self.m):
                        d_cur = (self._l[self._j[j]] - self._l[self._j[j - 1]]) - self.n_steps
                        if d_cur < d_min:
                            d_min, i_min = d_cur, j
                    i_min = 0 if d_min >= 0 else i_min
                    updated = self._j[i_min]
                    for j in range(i_min, self.m - 1):
                        self._j[j] = self._j[j + 1]
                    self._j[self.m - 1] = updated
            self._it = int(np.minimum(self.m, _n_generations + 1))
            self._l[self._j[self._it - 1]] = _n_generations*self.period
            pm[self._j[self._it - 1]] = p_c
            for i in range(0 if i_min == 1 else i_min, self._it):
                vm[self._j[i]] = self._a_inv_z(pm[self._j[i]], vm, d, i)
                v_n = np.dot(vm[self._j[i]], vm[self._j[i]])
                bd_3 = np.sqrt(1.0 + self._bd_2*v_n)
                b[self._j[i]] = self._a/v_n*(bd_3 - 1.0)
                d[self._j[i]] = self._c/v_n*(1.0 - 1.0/bd_3)
        if self._n_generations > 0:  # for PSR
            r = np.argsort(np.hstack((y, y_bak)))
            z_psr = np.sum(self._rr[r < self.n_individuals] - self._rr[r >= self.n_individuals])
            z_psr = z_psr/np.power(self.n_individuals, 2) - self.z_star
            s = (1 - self.c_s)*s + self.c_s*z_psr
            self.sigma *= np.exp(s/self.d_s)
        return mean_bak, p_c, s, vm, pm, b, d

    def restart_reinitialize(self, mean=None, x=None, p_c=None, s=None, vm=None,
                             pm=None, b=None, d=None, y=None):
        if ES.restart_reinitialize(self):
            mean, x, p_c, s, vm, pm, b, d, y = self.initialize(True)
        return mean, x, p_c, s, vm, pm, b, d, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        mean, x, p_c, s, vm, pm, b, d, y = self.initialize()
        while True:
            y_bak = np.copy(y)
            # sample and evaluate offspring population
            x, y = self.iterate(mean, x, pm, vm, y, b, args)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, p_c, s, vm, pm, b, d = self._update_distribution(
                mean, x, p_c, s, vm, pm, b, d, y, y_bak)
            self._print_verbose_info(y)
            self._n_generations += 1
            if self.is_restart:
                mean, x, p_c, s, vm, pm, b, d, y = self.restart_reinitialize(
                    mean, x, p_c, s, vm, pm, b, d, y)
        results = self._collect_results(fitness, mean)
        results['p_c'] = p_c
        results['s'] = s
        return results
