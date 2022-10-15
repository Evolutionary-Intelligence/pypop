import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class ES(Optimizer):
    """Evolution Strategies (ES).

    This is the **base** (abstract) class for all `ES` classes. Please use any of its instantiated subclasses to
    optimize the black-box problem at hand.

    .. note:: `ES` are a well-established family of randomized population-based search algorithms, proposed by two
       German computer scientists Ingo Rechenberg and Hans-Paul Schwefel (recipients of `Evolutionary Computation
       Pioneer Award 2002 <https://tinyurl.com/456as566>`_). One key property of `ES` is **adaptability of strategy
       parameters**, which generally can *significantly* accelerate the (local) convergence rate. Recently, the
       theoretical foundation of its most representative (modern) version called **CMA-ES** has been well built on
       the `Information-Geometric Optimization (IGO) <https://www.jmlr.org/papers/v18/14-467.html>`_ framework via
       invariance principles (inspired by `NES <https://jmlr.org/papers/v15/wierstra14a.html>`_).

       According to the latest `Nature <https://www.nature.com/articles/nature14544.>`_ review, *"the CMA-ES algorithm
       is widely regarded as the state of the art in numerical optimization"*.

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
                * 'n_individuals' - number of offspring (λ: lambda), offspring population size (`int`),
                * 'n_parents'     - number of parents (μ: mu), parental population size (`int`),
                * 'mean'          - initial (starting) point (`array_like`),
                * 'sigma'         - initial global step-size, mutation strength (`float`).

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring/descendants (λ: lambda), offspring population size.
    n_parents     : `int`
                    number of parents (μ: mu), parental population size.
    mean          : `array_like`
                    mean of Gaussian search/sampling/mutation distribution.
    sigma         : `float`
                    std of Gaussian search/sampling/mutation distribution, mutation strength.

    Methods
    -------

    References
    ----------
    https://homepages.fhv.at/hgb/downloads/ES-Is-Not-Gradient-Follower.pdf

    Ollivier, Y., Arnold, L., Auger, A. and Hansen, N., 2017.
    Information-geometric optimization algorithms: A unifying picture via invariance principles.
    Journal of Machine Learning Research, 18(18), pp.1-65.
    https://www.jmlr.org/papers/v18/14-467.html

    https://blog.otoro.net/2017/10/29/visual-evolution-strategies/

    Hansen, N., Arnold, D.V. and Auger, A., 2015.
    Evolution strategies.
    In Springer Handbook of Computational Intelligence (pp. 871-898).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007%2F978-3-662-43505-2_44

    http://www.scholarpedia.org/article/Evolution_strategies

    Beyer, H.G. and Schwefel, H.P., 2002.
    Evolution strategies–A comprehensive introduction.
    Natural Computing, 1(1), pp.3-52.
    https://link.springer.com/article/10.1023/A:1015059928466

    Rechenberg, I., 1989.
    Evolution strategy: Nature’s way of optimization.
    In Optimization: Methods and Applications, Possibilities and Limitations (pp. 106-126).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-642-83814-9_6

    Schwefel, H.P., 1984.
    Evolution strategies: A family of non-linear optimization techniques based on imitating
    some principles of organic evolution.
    Annals of Operations Research, 1(2), pp.165-167.
    https://link.springer.com/article/10.1007/BF01876146

    Rechenberg, I., 1984.
    The evolution strategy. A mathematical model of darwinian evolution.
    In Synergetics—from Microscopic to Macroscopic Order (pp. 122-132). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-642-69540-7_13
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # number of offspring (λ: lambda), offspring population size
            self.n_individuals = 4 + int(3*np.log(self.ndim_problem))  # only for small populations setting
        assert self.n_individuals > 0, f'`self.n_individuals` = {self.n_individuals}, but should > 0.'
        if self.n_parents is None:  # number of parents (μ: mu), parental population size
            self.n_parents = int(self.n_individuals/2)
            if self.n_parents > 1:
                # unify these settings in the base class ES for consistency and simplicity
                w_base, w = np.log((self.n_individuals + 1.0)/2.0), np.log(np.arange(self.n_parents) + 1.0)
                self._w = (w_base - w)/(self.n_parents*w_base - np.sum(w))
                self._mu_eff = 1.0/np.sum(np.power(self._w, 2))  # μ_eff / μ_w
                self._e_chi = np.sqrt(self.ndim_problem)*(  # E[||N(0,I)||]: expectation of chi distribution
                    1.0 - 1.0/(4.0*self.ndim_problem) + 1.0/(21.0*np.power(self.ndim_problem, 2)))
        assert self.n_parents > 0, f'`self.n_parents` = {self.n_parents}, but should > 0.'
        self.mean = options.get('mean')  # mean of Gaussian search/sampling/mutation distribution
        if self.mean is None:
            self.mean = options.get('x')
        self.sigma = options.get('sigma')  # global step-size (σ), mutation strength
        self.lr_mean = options.get('lr_mean')  # learning rate of mean update
        self.lr_sigma = options.get('lr_sigma')  # learning rate of sigma update
        # set options for restart
        self.sigma_threshold = options.get('sigma_threshold', 1e-12)  # stopping threshold of sigma
        self.stagnation = options.get('stagnation', int(10 + np.ceil(30*self.ndim_problem/self.n_individuals)))
        self.fitness_diff = options.get('fitness_diff', 1e-12)  # stopping threshold of fitness difference
        self._n_restart = 0  # only for restart
        self._sigma_bak = np.copy(self.sigma)  # only for restart
        self._fitness_list = [self.best_so_far_y]  # only for restart
        self._n_generations = 0  # number of generations

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _initialize_mean(self, is_restart=False):
        if is_restart or (self.mean is None):
            mean = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        else:
            mean = np.copy(self.mean)
        return mean

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def restart_reinitialize(self):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self.sigma = np.copy(self._sigma_bak)
            self.n_individuals *= 2
            self.n_parents = int(self.n_individuals/2)
            if self.n_parents > 1:
                w_base, w = np.log((self.n_individuals + 1.0)/2.0), np.log(np.arange(self.n_parents) + 1.0)
                self._w = (w_base - w)/(self.n_parents*w_base - np.sum(w))
                self._mu_eff = 1.0/np.sum(np.power(self._w, 2))  # μ_eff / μ_w
            self._n_restart += 1
            self._fitness_list = [np.Inf]
            self._n_generations = 0
            if self.verbose:
                print(' ....... restart .......')
        return is_restart

    def _collect_results(self, fitness=None, mean=None):
        results = Optimizer._collect_results(self, fitness)
        results['mean'] = mean
        results['sigma'] = self.sigma
        results['_n_restart'] = self._n_restart
        results['_n_generations'] = self._n_generations
        return results
