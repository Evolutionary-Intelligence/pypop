import numpy as np

from pypop7.optimizers.es.es import ES


class DDCMA(ES):
    """Diagonal Decoding Covariance Matrix Adaptation (DDCMA).

    .. note:: `DDCMA` is a *state-of-the-art* improvement version of the well-designed `CMA-ES` algorithm, which enjoys
       both two worlds of `SEP-CMA-ES` (faster adaptation on nearly separable problems) and `CMA-ES` (more robust
       adaptation on ill-conditioned non-separable problems) via **adaptive diagonal decoding**. It is **highly
       recommended** to first attempt other ES variants (e.g., `LMCMA`, `LMMAES`) for large-scale black-box
       optimization, since `DDCMA` has a *quadratic* time complexity (w.r.t. each sampling).

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
                  `4 + int(3*np.log(problem['ndim_problem']))`).

    Examples
    --------
    Use the optimizer `DDCMA` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.ddcma import DDCMA
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'is_restart': False,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> ddcma = DDCMA(problem, options)  # initialize the optimizer class
       >>> results = ddcma.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"DDCMA: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       DDCMA: 5000, 0.0

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/mc34kkmn>`_ for more details.

    Attributes
    ----------
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    sigma         : `float`
                    final global step-size, aka mutation strength.

    References
    ----------
    Akimoto, Y. and Hansen, N., 2020.
    Diagonal acceleration for covariance matrix adaptation evolution strategies.
    Evolutionary Computation, 28(3), pp.405-435.
    https://direct.mit.edu/evco/article/28/3/405/94999/Diagonal-Acceleration-for-Covariance-Matrix

    See its official Python implementation from Prof. Akimoto:
    https://gist.github.com/youheiakimoto/1180b67b5a0b1265c204cba991fa8518
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self._mu_eff = None
        self._mu_eff_negative = None
        self.c_s = None
        self.d_s = None
        self._gamma_s = 0.0
        self._gamma_c = 0.0
        self._gamma_d = 0.0
        self.c_1 = None
        self.c_w = None
        self.c_c = None
        self._w = None
        self.c_1_d = None
        self.c_w_d = None
        self.c_c_d = None
        self._w_d = None
        self._beta_eig = None
        self._t_eig = None
        self._n_generations = 0

    def _set_c_1_and_c_1_d(self, m):
        return 1.0/(2.0*(m/self.ndim_problem + 1.0)*np.power((self.ndim_problem + 1.0), 0.75) + self._mu_eff/2.0)

    def initialize(self, is_restart=False):
        w_apostrophe = np.log((self.n_individuals + 1.0)/2.0) - np.log(np.arange(self.n_individuals) + 1.0)
        positive_w, negative_w = w_apostrophe > 0, w_apostrophe < 0
        w_apostrophe[positive_w] /= np.sum(np.abs(w_apostrophe[positive_w]))
        w_apostrophe[negative_w] /= np.sum(np.abs(w_apostrophe[negative_w]))
        self._mu_eff = 1.0/np.sum(np.power(w_apostrophe[positive_w], 2))
        self._mu_eff_negative = 1.0/np.sum(np.power(w_apostrophe[negative_w], 2))
        self.c_s = (self._mu_eff + 2.0)/(self.ndim_problem + self._mu_eff + 5.0)
        self.d_s = 1.0 + self.c_s + 2.0*np.maximum(0.0, np.sqrt((self._mu_eff - 1.0)/(self.ndim_problem + 1.0)) - 1.0)
        mu_apostrophe = self._mu_eff + 1.0/self._mu_eff - 2.0 + self.n_individuals/(2.0*(self.n_individuals + 5.0))
        m = self.ndim_problem*(self.ndim_problem + 1)/2
        self._gamma_s = 0.0
        self._gamma_c = 0.0
        self._gamma_d = 0.0
        self.c_1 = self._set_c_1_and_c_1_d(m)
        self.c_w = np.minimum(mu_apostrophe*self.c_1, 1.0 - self.c_1)
        self.c_c = np.sqrt(self._mu_eff*self.c_1)/2.0
        self._w = np.copy(w_apostrophe)
        self._w[negative_w] *= np.minimum(1.0 + self.c_1/self.c_w, 1.0 + 2.0*self._mu_eff_negative/(self._mu_eff + 2.0))
        m = self.ndim_problem
        self.c_1_d = self._set_c_1_and_c_1_d(m)
        self.c_w_d = np.minimum(mu_apostrophe*self.c_1_d, 1.0 - self.c_1_d)
        self.c_c_d = np.sqrt(self._mu_eff*self.c_1_d)/2.0
        self._w_d = np.copy(w_apostrophe)
        self._w_d[negative_w] *= np.minimum(1.0 + self.c_1_d/self.c_w_d,
                                            1.0 + 2.0*self._mu_eff_negative/(self._mu_eff + 2.0))
        self._beta_eig = 10*self.ndim_problem
        self._t_eig = np.maximum(1.0, np.floor(1.0/(self._beta_eig*(self.c_1 + self.c_w))))
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        d = self.sigma*np.ones((self.ndim_problem,))  # diagonal decoding matrix
        self.sigma = 1.0
        sqrt_c = np.eye(self.ndim_problem)
        inv_sqrt_c = np.eye(self.ndim_problem)
        z = np.empty((self.n_individuals, self.ndim_problem))
        cz = np.empty((self.n_individuals, self.ndim_problem))
        x = np.empty((self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        p_s = np.zeros((self.ndim_problem,))
        p_c = np.zeros((self.ndim_problem,))
        p_c_d = np.zeros((self.ndim_problem,))
        cm = np.eye(self.ndim_problem)
        sqrt_eig_va = np.ones((self.ndim_problem,))
        self._list_initial_mean.append(np.copy(mean))
        self._n_generations = 0
        return mean, d, sqrt_c, inv_sqrt_c, z, cz, x, y, p_s, p_c, p_c_d, cm, sqrt_eig_va

    def iterate(self, mean=None, d=None, sqrt_c=None, z=None, cz=None, x=None, y=None, args=None):
        for k in range(self.n_individuals):  # to sample offspring population
            if self._check_terminations():
                return z, cz, x, y
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))  # Gaussian noise for mutation
            cz[k] = np.dot(z[k], sqrt_c)
            x[k] = mean + self.sigma*d*cz[k]  # offspring individual
            y[k] = self._evaluate_fitness(x[k], args)  # fitness
        self._n_generations += 1
        return z, cz, x, y

    def _update_distribution(self, mean=None, d=None, sqrt_c=None, inv_sqrt_c=None, z=None, cz=None,
                             x=None, y=None, p_s=None, p_c=None, p_c_d=None, cm=None, sqrt_eig_va=None):
        order = np.argsort(y)
        zz, xx = z[order], x[order]
        positive_w = self._w > 0
        wz = np.dot(self._w[positive_w], zz[positive_w])
        wcz = np.dot(self._w[positive_w], cz[order][positive_w])
        # update distribution mean via weighted multi-recombination
        mean += self.sigma*d*wcz
        # update global step-size via CSA
        p_s = (1.0 - self.c_s)*p_s + np.sqrt(self.c_s*(2.0 - self.c_s)*self._mu_eff)*wz
        self._gamma_s = np.power(1.0 - self.c_s, 2)*self._gamma_s + self.c_s*(2.0 - self.c_s)
        self.sigma *= np.exp(self.c_s/self.d_s*(np.linalg.norm(p_s)/self._e_chi - np.sqrt(self._gamma_s)))
        # update evolution path
        h_s = np.dot(p_s, p_s)/self._gamma_s < (2.0 + 4.0/(self.ndim_problem + 1.0))*self.ndim_problem
        p_c = (1.0 - self.c_c)*p_c + h_s*np.sqrt(self.c_c*(2.0 - self.c_c)*self._mu_eff)*d*wcz
        self._gamma_c = np.power(1.0 - self.c_c, 2)*self._gamma_c + h_s*self.c_c*(2.0 - self.c_c)
        # update covariance matrix
        pw, nw = self._w > 0, self._w < 0
        d_p = np.dot(p_c/d, inv_sqrt_c)
        cm += self.c_1*(np.outer(d_p, d_p) - self._gamma_c*np.eye(self.ndim_problem)) + self.c_w*(
                np.dot(np.transpose(zz[pw])*self._w[pw], zz[pw]) - np.sum(self._w[pw])*np.eye(self.ndim_problem) +
                np.dot(np.transpose(zz[nw])*(self._w[nw]*self.ndim_problem/np.power(np.linalg.norm(zz[nw], axis=1), 2)),
                       zz[nw]) - np.sum(self._w[nw])*np.eye(self.ndim_problem))  # Eq.19
        p_c_d = (1.0 - self.c_c_d)*p_c_d + h_s*np.sqrt(self.c_c_d*(2 - self.c_c_d)*self._mu_eff)*d*wcz
        self._gamma_d = np.power(1 - self.c_c_d, 2)*self._gamma_d + h_s*self.c_c_d*(2.0 - self.c_c_d)
        pwd, nwd = self._w_d > 0, self._w_d < 0
        eig_va = self.c_1_d*(np.power(np.dot(p_c_d/d, inv_sqrt_c), 2) - self._gamma_d) + self.c_w_d*(
                np.dot(self._w_d[pwd], np.power(zz[pwd], 2)) + (np.dot(self._w_d[nwd]*self.ndim_problem/np.power(
                    np.linalg.norm(zz[nwd], axis=1), 2), np.power(zz[nwd], 2)) - np.sum(self._w_d)))  # Eq.28
        d *= np.exp(eig_va/(2.0*np.maximum(1.0, np.max(sqrt_eig_va)/np.min(sqrt_eig_va) - 2.0 + 1.0)))  # Eq.29 + Eq.31
        if self._n_generations % self._t_eig == 0:
            c = np.dot(np.dot(sqrt_c, np.eye(self.ndim_problem) + np.minimum(
                0.75/np.abs(np.min(np.linalg.eigvalsh(cm))), 1.0)*cm), sqrt_c)  # Eq.21 + Eq.22
            sqrt_diag = np.sqrt(np.diag(c))
            d *= sqrt_diag  # Eq.34
            c = np.transpose(c/sqrt_diag)/sqrt_diag  # Eq.35 (correlation matrix)
            eig_va, eig_ve = np.linalg.eigh(c)  # to perform eigen decomposition
            sqrt_eig_va = np.sqrt(eig_va)
            sqrt_c = np.dot(eig_ve*sqrt_eig_va, np.transpose(eig_ve))
            inv_sqrt_c = np.dot(eig_ve/sqrt_eig_va, np.transpose(eig_ve))
            cm[:, :] = 0.0
        return mean, d, sqrt_c, inv_sqrt_c, cz, p_s, p_c, p_c_d, cm, sqrt_eig_va

    def restart_reinitialize(self, mean=None, d=None, sqrt_c=None, inv_sqrt_c=None, z=None, cz=None,
                             x=None, y=None, p_s=None, p_c=None, p_c_d=None, cm=None, sqrt_eig_va=None):
        if ES.restart_reinitialize(self, y):
            mean, d, sqrt_c, inv_sqrt_c, z, cz, x, y, p_s, p_c, p_c_d, cm, sqrt_eig_va = self.initialize(True)
        return mean, d, sqrt_c, inv_sqrt_c, z, cz, x, y, p_s, p_c, p_c_d, cm, sqrt_eig_va

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        mean, d, sqrt_c, inv_sqrt_c, z, cz, x, y, p_s, p_c, p_c_d, cm, sqrt_eig_va = self.initialize()
        while not self._check_terminations():
            # sample and evaluate offspring population
            z, cz, x, y = self.iterate(mean, d, sqrt_c, z, cz, x, y, args)
            mean, d, sqrt_c, inv_sqrt_c, cz, p_s, p_c, p_c_d, cm, sqrt_eig_va = self._update_distribution(
                mean, d, sqrt_c, inv_sqrt_c, z, cz, x, y, p_s, p_c, p_c_d, cm, sqrt_eig_va)
            self._print_verbose_info(fitness, y)
            if self.is_restart:
                mean, d, sqrt_c, inv_sqrt_c, z, cz, x, y, p_s, p_c, p_c_d, cm, sqrt_eig_va = self.restart_reinitialize(
                    mean, d, sqrt_c, inv_sqrt_c, z, cz, x, y, p_s, p_c, p_c_d, cm, sqrt_eig_va)
        results = self._collect(fitness, y, mean)
        # by default, do NOT save covariance matrix of search distribution in order to save memory,
        # owing to its *quadratic* space complexity
        return results
