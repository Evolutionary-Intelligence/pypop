import numpy as np

from pypop7.optimizers.es.es import ES


class VDCMA(ES):
    """Linear Covariance Matrix Adaptation (VDCMA).

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
                  `int(options['n_individuals']/2)`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.vdcma import VDCMA
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> vdcma = VDCMA(problem, options)  # initialize the optimizer class
       >>> results = vdcma.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"VDCMA: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       VDCMA: 5000, 7.116226375179302e-18

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/3e838zd5>`_ for more details.

    Attributes
    ----------
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.
    sigma         : `float`
                    final global step-size, aka mutation strength.

    References
    ----------
    Akimoto, Y., Auger, A. and Hansen, N., 2014, July.
    Comparison-based natural gradient optimization in high dimension.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 373-380). ACM.
    https://dl.acm.org/doi/abs/10.1145/2576768.2598258

    See the official Python version from Prof. Akimoto:
    https://gist.github.com/youheiakimoto/08b95b52dfbf8832afc71dfff3aed6c8
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.options = options
        self.c_factor = options.get('c_factor', np.maximum((self.ndim_problem - 5.0)/6.0, 0.5))
        self.c_c, self.c_1, self.c_mu, self.c_s = None, None, None, None
        self.d_s = None
        self._v_n, self._v_2, self._v_, self._v_p2 = None, None, None, None

    def initialize(self, is_restart=False):
        self.c_c = self.options.get('c_c', (4.0 + self._mu_eff/self.ndim_problem)/(
                self.ndim_problem + 4.0 + 2.0*self._mu_eff/self.ndim_problem))
        self.c_1 = self.options.get('c_1', self.c_factor*2.0/(
                np.square(self.ndim_problem + 1.3) + self._mu_eff))
        self.c_mu = self.options.get('c_mu', np.minimum(1.0 - self.c_1, self.c_factor*2.0*(
                self._mu_eff - 2.0 + 1.0/self._mu_eff)/(np.square(self.ndim_problem + 2.0) + self._mu_eff)))
        self.c_s = self.options.get('c_s', 1.0/(2.0*np.sqrt(self.ndim_problem/self._mu_eff) + 1.0))
        self.d_s = self.options.get('d_s', 1.0 + self.c_s + 2.0*np.maximum(0.0, np.sqrt(
            (self._mu_eff - 1.0)/(self.ndim_problem + 1.0)) - 1.0))
        d = np.ones((self.ndim_problem,))  # diagonal vector of sampling distribution
        # set principal search direction (vector) of sampling distribution
        v = self.rng_optimization.standard_normal((self.ndim_problem,))/np.sqrt(self.ndim_problem)
        p_s = np.zeros((self.ndim_problem,))  # evolution path for step-size adaptation (MCSA)
        p_c = np.zeros((self.ndim_problem,))  # evolution path for covariance matrix adaptation (CMA)
        self._v_n = np.linalg.norm(v)
        self._v_2 = np.square(self._v_n)
        self._v_ = v/self._v_n
        self._v_p2 = np.square(self._v_)
        z = np.empty((self.n_individuals, self.ndim_problem))  # Gaussian noise for mutation
        zz = np.empty((self.n_individuals, self.ndim_problem))  # search directions
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return d, v, p_s, p_c, z, zz, x, mean, y

    def iterate(self, d=None, z=None, zz=None, x=None, mean=None, y=None, args=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return z, zz, x, y
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            zz[k] = d*(z[k] + (np.sqrt(1.0 + self._v_2) - 1.0)*(np.dot(z[k], self._v_)*self._v_))
            x[k] = mean + self.sigma*zz[k]
            y[k] = self._evaluate_fitness(x[k], args)
        return z, zz, x, y

    def _p_q(self, zz, w=0):
        zz_v_ = np.dot(zz, self._v_)
        if isinstance(w, int) and w == 0:
            p = np.square(zz) - self._v_2/(1.0 + self._v_2)*(zz_v_*(zz*self._v_)) - 1.0
            q = zz_v_*zz - ((np.square(zz_v_) + 1.0 + self._v_2)/2.0)*self._v_
        else:
            p = np.dot(w, np.square(zz) - self._v_2/(1.0 + self._v_2)*(zz_v_*(zz*self._v_).T).T - 1.0)
            q = np.dot(w, (zz_v_*zz.T).T - np.outer((np.square(zz_v_) + 1.0 + self._v_2)/2.0, self._v_))
        return p, q

    def _update_distribution(self, d=None, v=None, p_s=None, p_c=None, zz=None, x=None, y=None):
        order = np.argsort(y)[:self.n_parents]
        # update mean
        mean = np.dot(self._w, x[order])
        # update global step-size
        z = np.dot(self._w, zz[order])/d
        z += (1.0/np.sqrt(1.0 + self._v_2) - 1.0)*np.dot(z, self._v_)*self._v_
        p_s = (1.0 - self.c_s)*p_s + np.sqrt(self.c_s*(2.0 - self.c_s)*self._mu_eff)*z
        p_s_2 = np.dot(p_s, p_s)
        self.sigma *= np.exp(self.c_s/self.d_s*(np.sqrt(p_s_2)/self._e_chi - 1.0))
        # update restricted covariance matrix (d, v)
        h_s = p_s_2 < (2.0 + 4.0/(self.ndim_problem + 1.0))*self.ndim_problem
        p_c = (1.0 - self.c_c)*p_c + h_s*np.sqrt(self.c_c*(2.0 - self.c_c)*self._mu_eff)*np.dot(self._w, zz[order])
        gamma = 1.0/np.sqrt(1.0 + self._v_2)
        alpha = np.sqrt(np.square(self._v_2) + (1.0 + self._v_2)/np.max(self._v_p2)*(2.0 - gamma))/(2.0 + self._v_2)
        if alpha < 1.0:
            beta = (4.0 - (2.0 - gamma)/np.max(self._v_p2))/np.square(1.0 + 2.0/self._v_2)
        else:
            alpha, beta = 1.0, 0.0
        b = 2.0*np.square(alpha) - beta
        a = 2.0 - (b + 2.0*np.square(alpha))*self._v_p2
        _v_p2_a = self._v_p2/a
        if self.c_mu == 0:
            p_mu, q_mu = np.zeros((self.ndim_problem,)), np.zeros((self.ndim_problem,))
        else:
            p_mu, q_mu = self._p_q(zz[order]/d, self._w)
        if self.c_1 == 0:
            p_1, q_1 = np.zeros((self.ndim_problem,)), np.zeros((self.ndim_problem,))
        else:
            p_1, q_1 = self._p_q(p_c/d)
        p = self.c_mu*p_mu + h_s*self.c_1*p_1
        q = self.c_mu*q_mu + h_s*self.c_1*q_1
        if self.c_mu + self.c_1 > 0:
            r = p - alpha/(1.0 + self._v_2)*((2.0 + self._v_2)*(
                    q*self._v_) - self._v_2*np.dot(self._v_, q)*self._v_p2)
            s = r/a - b*np.dot(r, _v_p2_a)/(1.0 + b*np.dot(self._v_p2, _v_p2_a))*_v_p2_a
            ng_v = q/self._v_n - alpha/self._v_n*((2.0 + self._v_2)*(
                    self._v_*s) - np.dot(s, np.square(self._v_))*self._v_)
            ng_d = d*s
            up_factor = np.minimum(np.minimum(1.0, 0.7*self._v_n/np.sqrt(np.dot(ng_v, ng_v))),
                                   0.7*(d/np.min(np.abs(ng_d))))
        else:
            ng_v, ng_d, up_factor = np.zeros((self.ndim_problem,)), np.zeros((self.ndim_problem,)), 1.0
        v += up_factor*ng_v
        d += up_factor*ng_d
        self._v_n = np.linalg.norm(v)
        self._v_2 = np.square(self._v_n)
        self._v_ = v/self._v_n
        self._v_p2 = np.square(self._v_)
        return mean, p_s, p_c, v, d

    def restart_reinitialize(self, d=None, v=None, p_s=None, p_c=None, z=None, zz=None, x=None, mean=None, y=None):
        if self.is_restart and ES.restart_reinitialize(self, y):
            d, v, p_s, p_c, z, zz, x, mean, y = self.initialize(True)
        return d, v, p_s, p_c, z, zz, x, mean, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        d, v, p_s, p_c, z, zz, x, mean, y = self.initialize()
        while not self.termination_signal:
            # sample and evaluate offspring population
            z, zz, x, y = self.iterate(d, z, zz, x, mean, y, args)
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            mean, p_s, p_c, v, d = self._update_distribution(d, v, p_s, p_c, zz, x, y)
            self._n_generations += 1
            d, v, p_s, p_c, z, zz, x, mean, y = self.restart_reinitialize(
                d, v, p_s, p_c, z, zz, x, mean, y)
        results = self._collect(fitness, y, mean)
        results['d'] = d
        results['v'] = v
        return results
