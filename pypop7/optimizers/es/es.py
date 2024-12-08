import numpy as np  # engine for numerical computing

# abstract class of all Black-Box Optimizers (BBO)
from pypop7.optimizers.core.optimizer import Optimizer


class ES(Optimizer):
    """Evolution Strategies (ES).

    This is the **abstract** class for all `ES` classes. Please use any of instantiated
    subclasses of `ES` to optimize the black-box problem at hand.

    .. note:: `ES` are a well-established family of randomized **population-based**
       search algorithms, proposed by two German students Ingo Rechenberg and
       Hans-Paul Schwefel (two recipients of `IEEE Evolutionary Computation Pioneer
       Award 2002 <https://tinyurl.com/456as566>`_). One key property of `ES` is
       **adaptability of strategy parameters**, which can *significantly* accelerate
       the (local) convergence rate in many cases. Recently, the **theoretical
       foundation** of its most representative (modern) version **CMA-ES** has been in
       part built on the `Information-Geometric Optimization (IGO)
       <https://www.jmlr.org/papers/v18/14-467.html>`_ framework via **invariance**
       principles (**IGO** was originally inspired by `Natural Evolution Strategies 
       <https://jmlr.org/papers/v15/wierstra14a.html>`_).

       `"Their two most prominent design principles are unbiasedness and adaptive
       control of parameters of the sample distribution. Parameter control is not
       always directly inspired by biological evolution, but is an indispensable
       and central feature of ES. The result of selection and recombination is often
       deterministic. For recombination, using a single parental centroid has become
       the most popular approach, because such algorithms are simpler to formalize,
       easier to analyze, and even perform better in various circumstances as they
       allow for maximum genetic repair."---[Hansen et al., 2015]
       <https://link.springer.com/chapter/10.1007%2F978-3-662-43505-2_44>`_

       In 2017, `OpenAI <https://openai.com/>`_ designed a **scalable** `ES` version
       (called `OpenAI-ES <https://openai.com/index/evolution-strategies/>`_) which
       obtained *competitive* performance for deep neural network-based policy
       search in reinforcement learning.

       For some interesting applications of `ES`, please refer to
       `[SIMULIA > CST Studio Suite > Automatic Optimization (Dassault Systèmes)]
       <https://www.3ds.com/products/simulia/cst-studio-suite/automatic-optimization>`_,
       `[Yang et al., 2024, CVPR (CMU)]
       <https://diffusion-es.github.io/>`_,
       `[Elfikky et al., 2024, LWC (UCSC + Qualcomm Inc.)]
       <https://ieeexplore.ieee.org/abstract/document/10531788>`_,
       `[Martin&Sandholm, 2024 (CMU + Strategy Robot, Inc. + Optimized Markets, Inc. + Strategic Machine, Inc.)]
       <https://arxiv.org/abs/2406.08687>`_,
       `[Reali et al., 2024 (Microsoft Research, Gates Medical Research Institute, Hackensack Meridian Health, etc.)]
       <https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2023.1272091/full>`_
       `[Lee et al., 2023, Science Robotics (UMich + Google X + Google Robotics + Georgia Tech)]
       <https://www.science.org/doi/10.1126/scirobotics.adg3705>`_,
       `[Milekovic et al., 2023, Nature Medicine (EPFL + ONWARD Medical + Medtronic Inc. + Oxford, etc.)]
       <https://www.nature.com/articles/s41591-023-02584-1>`_,
       `[Koginov et al., 2023, TMRB]
       <https://ieeexplore.ieee.org/document/10302449>`_, `[Sun et al., 2023, ACL]
       <https://aclanthology.org/2023.acl-long.625.pdf>`_,
       `[NeurIPS-Workshop-2023] <https://openreview.net/pdf?id=KiQ8PhDzIU>`_,
       `[TVCG-2014] <https://ieeexplore.ieee.org/abstract/document/6781622>`_,
       `[AIAAJ-2014] <https://arc.aiaa.org/doi/full/10.2514/1.J052715>`_,
       `[ICML-2009] <https://dl.acm.org/doi/abs/10.1145/1553374.1553426>`_,
       `[Nature-2000] <https://www.nature.com/articles/35023115>`_,
       just to name a few.

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
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - number of offspring/descendants, aka offspring population size (`int`),
                * 'n_parents'     - number of parents/ancestors, aka parental population size (`int`),
                * 'mean'          - initial (starting) point (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'sigma'         - initial global step-size, aka mutation strength (`float`).

    Attributes
    ----------
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search/sampling/mutation distribution.
    n_individuals : `int`
                    number of offspring/descendants, aka offspring population size.
    n_parents     : `int`
                    number of parents/ancestors, aka parental population size.
    sigma         : `float`
                    global step-size, aka mutation strength (i.e., overall std of Gaussian search distribution).

    Methods
    -------

    References
    ----------
    https://homepages.fhv.at/hgb/downloads/ES-Is-Not-Gradient-Follower.pdf

    Ollivier, Y., Arnold, L., Auger, A. and Hansen, N., 2017.
    `Information-geometric optimization algorithms: A unifying picture via invariance principles.
    <https://www.jmlr.org/papers/v18/14-467.html>`_
    Journal of Machine Learning Research, 18(18), pp.1-65.

    https://blog.otoro.net/2017/10/29/visual-evolution-strategies/

    Hansen, N., Arnold, D.V. and Auger, A., 2015.
    `Evolution strategies.
    <https://link.springer.com/chapter/10.1007%2F978-3-662-43505-2_44>`_
    In Springer Handbook of Computational Intelligence (pp. 871-898).
    Springer, Berlin, Heidelberg.

    Bäck, T., Foussette, C., & Krause, P. (2013).
    `Contemporary evolution strategies.
    <https://link.springer.com/book/10.1007/978-3-642-40137-4>`_
    Berlin: Springer.

    http://www.scholarpedia.org/article/Evolution_strategies

    Beyer, H.G. and Schwefel, H.P., 2002.
    `Evolution strategies–A comprehensive introduction.
    <https://link.springer.com/article/10.1023/A:1015059928466>`_
    Natural Computing, 1(1), pp.3-52.

    Rechenberg, I., 2000.
    `Case studies in evolutionary experimentation and computation.
    <https://www.sciencedirect.com/science/article/abs/pii/S0045782599003813>`_
    Computer Methods in Applied Mechanics and Engineering, 186(2-4), pp.125-140.

    Rechenberg, I., 1989.
    `Evolution strategy: Nature’s way of optimization.
    <https://link.springer.com/chapter/10.1007/978-3-642-83814-9_6>`_
    In Optimization: Methods and Applications, Possibilities and Limitations (pp. 106-126).
    Springer, Berlin, Heidelberg.

    Schwefel, H.P., 1988.
    `Collective intelligence in evolving systems.
    <https://link.springer.com/chapter/10.1007/978-3-642-73953-8_8>`_
    In Ecodynamics (pp. 95-100). Springer, Berlin, Heidelberg.

    Schwefel, H.P., 1984.
    `Evolution strategies: A family of non-linear optimization techniques based on imitating
    some principles of organic evolution.
    <https://link.springer.com/article/10.1007/BF01876146>`_
    Annals of Operations Research, 1(2), pp.165-167.

    Rechenberg, I., 1984.
    `The evolution strategy. A mathematical model of darwinian evolution.
    <https://link.springer.com/chapter/10.1007/978-3-642-69540-7_13>`_
    In Synergetics—from Microscopic to Macroscopic Order (pp. 122-132). Springer, Berlin, Heidelberg.
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        # population size, sample size, number of offspring (Nikolaus Hansen, 2023)
        if self.n_individuals is None:  # number of offspring (λ: lambda), offspring population size
            self.n_individuals = 4 + int(3*np.log(self.ndim_problem))  # only for small populations setting
        assert self.n_individuals > 0, f'`self.n_individuals` = {self.n_individuals}, but should > 0.'
        # parent number, number of (positively) selected search points in the population,
        #   number of strictly positive recombination weights (Nikolaus Hansen, 2023)
        if self.n_parents is None:  # number of parents (μ: mu), parental population size
            self.n_parents = int(self.n_individuals/2)
        assert self.n_parents <= self.n_individuals,\
            f'self.n_parents (== {self.n_parents}) should <= self.n_individuals (== {self.n_individuals})'
        if self.n_parents > 1:
            self._w, self._mu_eff = self._compute_weights()
            self._e_chi = np.sqrt(self.ndim_problem)*(  # E[||N(0,I)||]: expectation of chi distribution
                1.0 - 1.0/(4.0*self.ndim_problem) + 1.0/(21.0*np.square(self.ndim_problem)))
        assert self.n_parents > 0, f'`self.n_parents` = {self.n_parents}, but should > 0.'
        self.mean = options.get('mean')  # mean of Gaussian search/sampling/mutation distribution
        if self.mean is None:  # `mean` overwrites `x` if both are set
            self.mean = options.get('x')
        # "overall" standard deviation, mutation strength (Nikolaus Hansen, 2023; Hans-Georg Beyer, 2017)
        self.sigma = options.get('sigma')  # global step-size (σ)
        assert self.sigma > 0, f'`self.sigma` = {self.sigma}, but should > 0.'
        self.lr_mean = options.get('lr_mean')  # learning rate of mean update
        assert self.lr_mean is None or self.lr_mean > 0,\
            f'`self.lr_mean` = {self.lr_mean}, but should > 0.'
        self.lr_sigma = options.get('lr_sigma')  # learning rate of sigma update
        assert self.lr_sigma is None or self.lr_sigma > 0,\
            f'`self.lr_sigma` = {self.lr_sigma}, but should > 0.'
        self._printed_evaluations = self.n_function_evaluations
        # set options for *restart*
        self._n_restart = 0  # only for restart
        self._n_generations = 0  # number of generations
        self._list_generations = []  # list of number of generations for all restarts
        self._list_fitness = [self.best_so_far_y]  # only for restart
        self._list_initial_mean = []  # list of mean for each restart
        self.sigma_threshold = options.get('sigma_threshold', 1e-12)  # stopping threshold of sigma
        self.stagnation = options.get('stagnation', int(10 + np.ceil(30*self.ndim_problem/self.n_individuals)))
        self.fitness_diff = options.get('fitness_diff', 1e-12)  # stopping threshold of fitness difference
        self._sigma_bak = np.copy(self.sigma)  # initial global step-size -> only for restart

    def _compute_weights(self):
        # unify these following settings in the base class for *consistency* and *simplicity*
        w_base, w = np.log((self.n_individuals + 1.0)/2.0), np.log(np.arange(self.n_parents) + 1.0)
        # positive weight coefficients for weighted intermediate recombination (Nikolaus Hansen, 2023)
        #   [assigning different weights should be interpreted as a selection mechanism]
        w = (w_base - w)/(self.n_parents*w_base - np.sum(w))
        # variance effective selection mass (Nikolaus Hansen, 2023)
        #   effective sample size of the selected samples
        mu_eff = 1.0/np.sum(np.square(w))  # μ_eff (μ_w)
        return w, mu_eff

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _initialize_mean(self, is_restart=False):
        if is_restart or (self.mean is None):
            mean = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        else:
            mean = np.copy(self.mean)
        self.mean = np.copy(mean)
        return mean

    def _print_verbose_info(self, fitness, y, is_print=False):
        if y is not None:
            if self.saving_fitness:
                if not np.isscalar(y):
                    fitness.extend(y)
                else:
                    fitness.append(y)
        if self.verbose:
            is_verbose = self._printed_evaluations != self.n_function_evaluations  # to avoid repeated printing
            is_verbose_1 = (not self._n_generations % self.verbose) and is_verbose
            is_verbose_2 = self.termination_signal > 0 and is_verbose
            is_verbose_3 = is_print and is_verbose
            if is_verbose_1 or is_verbose_2 or is_verbose_3:
                info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
                print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))
                self._printed_evaluations = self.n_function_evaluations

    def restart_reinitialize(self, y):
        min_y = np.min(y)
        if min_y < self._list_fitness[-1]:
            self._list_fitness.append(min_y)
        else:
            self._list_fitness.append(self._list_fitness[-1])
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._list_fitness) >= self.stagnation:
            is_restart_2 = (self._list_fitness[-self.stagnation] - self._list_fitness[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self._print_verbose_info([], y, True)
            if self.verbose:
                print(' ....... *** restart *** .......')
            self._n_restart += 1
            self._list_generations.append(self._n_generations)  # for each restart
            self._n_generations = 0
            self._list_fitness = [np.inf]
            self.sigma = np.copy(self._sigma_bak)
            self.n_individuals *= 2
            self.n_parents = int(self.n_individuals/2)
            if self.n_parents > 1:
                self._w, self._mu_eff = self._compute_weights()
        return is_restart

    def _collect(self, fitness=None, y=None, mean=None):
        self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results['mean'] = mean  # final mean of search distribution
        results['initial_mean'] = self.mean  # initial mean of search distribution
        results['_list_initial_mean'] = self._list_initial_mean  # list of initial mean for each restart
        # by default, do NOT save covariance matrix of search distribution in order to save memory,
        # owing to its *quadratic* space complexity
        results['sigma'] = self.sigma  # only global step-size of search distribution
        results['_n_restart'] = self._n_restart  # number of restart
        results['_n_generations'] = self._n_generations  # number of generations
        results['_list_generations'] = self._list_generations  # list of number of generations for each restart
        return results
