import numpy as np  # engine for numerical computing

from pypop7.optimizers.es.es import ES


class CMAES(ES):
    """Covariance Matrix Adaptation Evolution Strategy (CMAES).

    .. note:: `CMAES` is widely recognized as one of **State Of The Art (SOTA)** evolutionary algorithms for black-box
       optimization, according to the `Nature <https://www.nature.com/articles/nature14544>`_ review of Evolutionary
       Computation. Currently, the standard `(µ/µ_W, λ)-CMA-ES` version is considered in our Python implementation.

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
    Use the optimizer `CMAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.cmaes import CMAES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'is_restart': False,
       ...            'mean': 3.0*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> cmaes = CMAES(problem, options)  # initialize the optimizer class
       >>> results = cmaes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CMAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CMAES: 5000, 9.11305771685916e-09

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/4mysrjwe>`_ for more details.

    Attributes
    ----------
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size / sample size.
    n_parents     : `int`
                    number of parents, aka parental population size / number of positively selected search points.
    sigma         : `float`
                    final global step-size, aka mutation strength.

    References
    ----------
    Hansen, N., 2023.
    The CMA evolution strategy: A tutorial.
    arXiv preprint arXiv:1604.00772.
    https://arxiv.org/abs/1604.00772

    Ollivier, Y., Arnold, L., Auger, A. and Hansen, N., 2017.
    Information-geometric optimization algorithms: A unifying picture via invariance principles.
    Journal of Machine Learning Research, 18(18), pp.1-65.
    https://jmlr.org/papers/v18/14-467.html

    Hansen, N., Müller, S.D. and Koumoutsakos, P., 2003.
    Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES).
    Evolutionary Computation, 11(1), pp.1-18.
    https://direct.mit.edu/evco/article-abstract/11/1/1/1139/Reducing-the-Time-Complexity-of-the-Derandomized

    Hansen, N. and Ostermeier, A., 2001.
    Completely derandomized self-adaptation in evolution strategies.
    Evolutionary Computation, 9(2), pp.159-195.
    https://direct.mit.edu/evco/article-abstract/9/2/159/892/Completely-Derandomized-Self-Adaptation-in

    Hansen, N. and Ostermeier, A., 1996, May.
    Adapting arbitrary normal mutation distributions in evolution strategies: The covariance matrix adaptation.
    In Proceedings of IEEE International Conference on Evolutionary Computation (pp. 312-317). IEEE.
    https://ieeexplore.ieee.org/abstract/document/542381

    See one *lightweight* Python implementation of CMA-ES from `cyberagent.ai`:
    https://github.com/CyberAgentAILab/cmaes
    """
    def __init__(self, problem, options):
        self.options = options
        ES.__init__(self, problem, options)
        assert self.n_individuals >= 2
        self._w, self._mu_eff, self._mu_eff_minus = None, None, None  # variance effective selection mass
        self.c_s, self.d_sigma = None, None  # for cumulative step-length adaptation (CSA)
        self._p_s_1, self._p_s_2 = None, None  # for evolution path update of CSA
        self._p_c_1, self._p_c_2 = None, None  # for evolution path update of CMA
        # c_1 -> learning rate for the rank-one update of CMA
        # c_w (c_μ) -> learning rate for the rank-µ update of CMA
        self.c_c, self.c_1, self.c_w, self._alpha_cov = None, None, None, 2.0  # for CMA (c_w -> c_μ)

    def _set_c_c(self):  # to set decay rate of evolution path for rank-one update of CMA
        return (4.0 + self._mu_eff/self.ndim_problem)/(self.ndim_problem + 4.0 + 2.0*self._mu_eff/self.ndim_problem)

    def _set_c_w(self):
        return np.minimum(1.0 - self.c_1, self._alpha_cov*(1.0/4.0 + self._mu_eff + 1.0/self._mu_eff - 2.0) /
                          (np.square(self.ndim_problem + 2.0) + self._alpha_cov*self._mu_eff/2.0))

    def _set_d_sigma(self):
        return 1.0 + 2.0*np.maximum(0.0, np.sqrt((self._mu_eff - 1.0)/(self.ndim_problem + 1.0)) - 1.0) + self.c_s

    def initialize(self, is_restart=False):
        w_a = np.log((self.n_individuals + 1.0)/2.0) - np.log(np.arange(self.n_individuals) + 1.0)  # w_apostrophe
        self._mu_eff = np.square(np.sum(w_a[:self.n_parents]))/np.sum(np.square(w_a[:self.n_parents]))
        self._mu_eff_minus = np.square(np.sum(w_a[self.n_parents:]))/np.sum(np.square(w_a[self.n_parents:]))
        self.c_s = self.options.get('c_s', (self._mu_eff + 2.0)/(self.ndim_problem + self._mu_eff + 5.0))
        self.d_sigma = self.options.get('d_sigma', self._set_d_sigma())
        self.c_c = self.options.get('c_c', self._set_c_c())
        self.c_1 = self.options.get('c_1', self._alpha_cov/(np.square(self.ndim_problem + 1.3) + self._mu_eff))
        self.c_w = self.options.get('c_w', self._set_c_w())
        w_min = np.min([1.0 + self.c_1/self.c_w, 1.0 + 2.0*self._mu_eff_minus/(self._mu_eff + 2.0),
                        (1.0 - self.c_1 - self.c_w)/(self.ndim_problem*self.c_w)])
        self._w = np.where(w_a >= 0, 1.0/np.sum(w_a[w_a > 0])*w_a, w_min/(-np.sum(w_a[w_a < 0]))*w_a)
        self._p_s_1, self._p_s_2 = 1.0 - self.c_s, np.sqrt(self.c_s*(2.0 - self.c_s)*self._mu_eff)
        self._p_c_1, self._p_c_2 = 1.0 - self.c_c, np.sqrt(self.c_c*(2.0 - self.c_c)*self._mu_eff)
        x = np.empty((self.n_individuals, self.ndim_problem))  # a population of search points (individuals, offspring)
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        p_s = np.zeros((self.ndim_problem,))  # evolution path (p_σ) for cumulative step-length adaptation (CSA)
        p_c = np.zeros((self.ndim_problem,))  # evolution path for covariance matrix adaptation (CMA)
        cm = np.eye(self.ndim_problem)  # covariance matrix of Gaussian search distribution
        e_ve = np.eye(self.ndim_problem)  # eigenvectors of `cm` (orthogonal matrix)
        e_va = np.ones((self.ndim_problem,))  # square roots of eigenvalues of `cm` (in diagonal rather matrix form)
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        d = np.empty((self.n_individuals, self.ndim_problem))
        self._list_initial_mean.append(np.copy(mean))
        return x, mean, p_s, p_c, cm, e_ve, e_va, y, d

    def iterate(self, x=None, mean=None, e_ve=None, e_va=None, y=None, d=None, args=None):
        for k in range(self.n_individuals):  # to sample offspring population
            if self._check_terminations():
                return x, y, d
            # produce a spherical (isotropic) Gaussian distribution (Nikolaus Hansen, 2023)
            z = self.rng_optimization.standard_normal((self.ndim_problem,))  # Gaussian noise for mutation
            d[k] = np.dot(e_ve @ np.diag(e_va), z)
            x[k] = mean + self.sigma*d[k]  # offspring individual
            y[k] = self._evaluate_fitness(x[k], args)  # fitness
        return x, y, d

    def update_distribution(self, x=None, p_s=None, p_c=None, cm=None, e_ve=None, e_va=None, y=None, d=None):
        order = np.argsort(y)  # to rank all offspring individuals
        wd = np.dot(self._w[:self.n_parents], d[order[:self.n_parents]])
        # update distribution mean via weighted recombination
        mean = np.dot(self._w[:self.n_parents], x[order[:self.n_parents]])
        # update global step-size (CSA)
        cm_minus_half = e_ve @ np.diag(1.0/e_va) @ e_ve.T
        p_s = self._p_s_1*p_s + self._p_s_2*np.dot(cm_minus_half, wd)
        self.sigma *= np.exp(self.c_s/self.d_sigma*(np.linalg.norm(p_s)/self._e_chi - 1.0))
        # update covariance matrix (CMA)
        h_s = 1.0 if np.linalg.norm(p_s)/np.sqrt(1.0 - np.power(1.0 - self.c_s, 2*(self._n_generations + 1))) < (
                1.4 + 2.0/(self.ndim_problem + 1.0))*self._e_chi else 0.0
        p_c = self._p_c_1*p_c + h_s*self._p_c_2*wd
        w_o = self._w*np.where(self._w >= 0, 1.0, self.ndim_problem/(np.square(
            np.linalg.norm(cm_minus_half @ d.T, axis=0)) + 1e-8))
        cm = ((1.0 + self.c_1*(1.0 - h_s)*self.c_c*(2.0 - self.c_c) - self.c_1 - self.c_w*np.sum(self._w))*cm +
              self.c_1*np.outer(p_c, p_c))  # rank-one update
        for i in range(self.n_individuals):  # rank-μ update (to estimate variances of sampled *steps*)
            cm += self.c_w*w_o[i]*np.outer(d[order[i]], d[order[i]])
        # do eigen-decomposition and return both eigenvalues and eigenvectors
        cm = (cm + np.transpose(cm))/2.0  # to ensure symmetry of covariance matrix
        # use `np.linalg.eigh` rather than `np.linalg.eig` according to
        #   https://stackoverflow.com/questions/45434989/numpy-difference-between-linalg-eig-and-linalg-eigh
        e_va, e_ve = np.linalg.eigh(cm)  # e_va -> eigenvalues, e_ve -> eigenvectors
        e_va = np.sqrt(np.where(e_va < 0.0, 1e-8, e_va))  # to avoid negative eigenvalues
        # e_va: squared root of eigenvalues -> interpreted as individual step-sizes and its diagonal entries are
        #   standard deviations of different components (Nikolaus Hansen, 2023)
        cm = e_ve @ np.diag(np.square(e_va)) @ np.transpose(e_ve)  # to recover covariance matrix
        return mean, p_s, p_c, cm, e_ve, e_va

    def restart_reinitialize(self, x=None, mean=None, p_s=None, p_c=None,
                             cm=None, e_ve=None, e_va=None, y=None, d=None):
        if ES.restart_reinitialize(self, y):
            x, mean, p_s, p_c, cm, e_ve, e_va, y, d = self.initialize(True)
        return x, mean, p_s, p_c, cm, e_ve, e_va, y, d

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p_s, p_c, cm, e_ve, e_va, y, d = self.initialize()
        while True:
            # sample and evaluate offspring population
            x, y, d = self.iterate(x, mean, e_ve, e_va, y, d, args)
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            mean, p_s, p_c, cm, e_ve, e_va = self.update_distribution(x, p_s, p_c, cm, e_ve, e_va, y, d)
            if self.is_restart:
                x, mean, p_s, p_c, cm, e_ve, e_va, y, d = self.restart_reinitialize(
                    x, mean, p_s, p_c, cm, e_ve, e_va, y, d)
        results = self._collect(fitness, y, mean)
        results['p_s'] = p_s
        results['p_c'] = p_c
        results['e_va'] = e_va
        # results['e_ve'] = e_ve  # do NOT save covariance matrix, owing to its *quadratic* space complexity
        return results
