import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class ES(Optimizer):
    """Evolution Strategies (ES).

    This is the **base** (abstract) class for all ES classes. Please use any of its concrete subclasses to
    optimize the black-box problem at hand.

    .. note:: Its three methods (`initialize`, `iterate`, `optimize`) should be implemented by its subclasses.

       `ES` are a well-established family of randomized population-based search algorithms, proposed originally by two
       German computer scientists Rechenberg and Schwefel (recipients of `Evolutionary Computation Pioneer Award 2002
       <https://tinyurl.com/456as566>`_). One key property of `ES` lies in its adaptability of strategy parameters,
       which can *significantly* accelerate the (local) convergence rate in many cases. Recently, the theoretical
       foundation of its most representative (modern) version called **CMA-ES** has been built on the very interesting
       `Information-Geometric Optimization (IGO) <https://www.jmlr.org/papers/v18/14-467.html>`_ framework via
       invariance principles.

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
                * 'max_runtime'              - maximal runtime (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`),
                * 'record_fitness'           - flag to record fitness list to output results (`bool`, default: `False`),
                * 'record_fitness_frequency' - function evaluations frequency of recording (`int`, default: `1000`),

                  * if `record_fitness` is set to `False`, it will be ignored,
                  * if `record_fitness` is set to `True` and it is set to 1, all fitness generated during optimization
                    will be saved into output results.

                * 'verbose'                  - flag to print verbose info during optimization (`bool`, default: `True`),
                * 'verbose_frequency'        - frequency of printing verbose info (`int`, default: `10`);
              and with four particular settings (`keys`):
                * 'n_individuals' - number of offspring (λ: lambda), offspring population size (`int`),
                * 'n_parents'     - number of parents (μ: mu), parental population size (`int`),
                * 'mean'          - initial (starting) point, mean of Gaussian search distribution (`array_like`),
                * 'sigma'         - initial global step-size (σ), mutation strength (`float`).

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring (λ: lambda), offspring population size.
    n_parents     : `int`
                    number of parents (μ: mu), parental population size.
    mean          : `array_like`
                    initial (starting) point, mean of Gaussian search distribution.
    sigma         : `float`
                    initial global step-size (σ), mutation strength (`float`).

    Methods
    -------

    References
    ----------
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
            self.n_individuals = 4 + int(3 * np.log(self.ndim_problem))  # for small populations setting
        assert self.n_individuals > 0, f'`self.n_individuals` = {self.n_individuals}, but should > 0.'
        if self.n_parents is None:  # number of parents (μ: mu), parental population size
            self.n_parents = int(self.n_individuals / 2)
            if self.n_parents > 1:
                # for consistency and simplicity, we unify these in the base class ES.
                w_base, w = np.log((self.n_individuals + 1) / 2), np.log(np.arange(self.n_parents) + 1)
                self._w = (w_base - w) / (self.n_parents * w_base - np.sum(w))
                self._mu_eff = 1 / np.sum(np.power(self._w, 2))  # μ_eff / μ_w
                # E[||N(0,I)||]: expectation of chi distribution
                self._e_chi = np.sqrt(self.ndim_problem) * (
                        1 - 1 / (4 * self.ndim_problem) + 1 / (21 * np.power(self.ndim_problem, 2)))
        assert self.n_parents > 0, f'`` = {self.n_parents}, but should > 0.'
        self.mean = options.get('mean')  # mean of Gaussian search distribution
        if self.mean is None:  # 'mean' has priority over 'x'
            self.mean = options.get('x')
        self.sigma = options.get('sigma')  # global step-size (σ)
        self.eta_mean = options.get('eta_mean')  # learning rate of mean
        self.eta_sigma = options.get('eta_sigma')  # learning rate of std
        self._n_generations = 0
        # for restart
        self.n_restart = 0
        self.sigma_threshold = options.get('sigma_threshold', 1e-10)  # stopping threshold of sigma for restart
        # maximal generation number of fitness stagnation for restart
        self.stagnation = options.get('stagnation', np.maximum(32, self.ndim_problem))
        self.fitness_diff = options.get('fitness_diff', 1e-10)  # stopping threshold of fitness difference for restart
        self._sigma_bak = np.copy(self.sigma)  # bak for restart
        self._fitness_list = [self.best_so_far_y]  # to store `best_so_far_y` generated in each generation

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
        if self.verbose and (not self._n_generations % self.verbose_frequency):
            best_so_far_y = -self.best_so_far_y if self._is_maximization else self.best_so_far_y
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, best_so_far_y, np.min(y), self.n_function_evaluations))

    def restart_initialize(self):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self.n_restart += 1
            self.sigma = np.copy(self._sigma_bak)
            self.n_individuals *= 2
            self.n_parents = int(self.n_individuals / 2)
            if self.n_parents > 1:
                w_base, w = np.log((self.n_individuals + 1) / 2), np.log(np.arange(self.n_parents) + 1)
                self._w = (w_base - w) / (self.n_parents * w_base - np.sum(w))
                self._mu_eff = 1 / np.sum(np.power(self._w, 2))
            self._n_generations = 0
            self._fitness_list = [np.Inf]
        return is_restart

    def _collect_results(self, fitness, mean=None):
        results = Optimizer._collect_results(self, fitness)
        results['mean'] = mean
        results['sigma'] = self.sigma
        results['n_restart'] = self.n_restart
        results['_n_generations'] = self._n_generations
        return results
