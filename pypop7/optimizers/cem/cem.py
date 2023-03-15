import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class CEM(Optimizer):
    """Cross-Entropy Method (CEM).

    This is the **abstract** class for all `CEM` classes. Please use any of its instantiated subclasses to
    optimize the black-box problem at hand.

    .. note:: `CEM` is a class of **principled** population-based optimizers, proposed originally by *Rubinstein*,
        whose core idea is based on **Kullback–Leibler (or Cross-Entropy) minimization**.

       `"CEM is not only based on fundamental principles (cross-entropy distance, maximum likelihood, etc.), but is
       also very easy to program (with far fewer parameters than many other global optimization heuristics), and
       gives consistently accurate results, and is therefore worth considering when faced with a difficult
       optimization problem."---[Kroese et al., 2006, MCAP]
       <https://link.springer.com/article/10.1007/s11009-006-9753-0>`_

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

                * 'n_individuals' - number of individuals/samples (`int`, default: `1000`),
                * 'n_parents'     - number of elitists (`int`, default: `200`),

    Attributes
    ----------
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search (mutation/sampling) distribution.
    n_individuals : `int`
                    number of individuals/samples.
    n_parents     : `int`
                    number of elitists.
    sigma         : `float`
                    initial global step-size, aka mutation strength.

    Methods
    -------

    References
    ----------
    Amos, B. and Yarats, D., 2020, November.
    The differentiable cross-entropy method.
    In International Conference on Machine Learning (pp. 291-302). PMLR.
    http://proceedings.mlr.press/v119/amos20a.html

    Rubinstein, R.Y. and Kroese, D.P., 2016.
    Simulation and the Monte Carlo method (Third Edition).
    John Wiley & Sons.
    https://onlinelibrary.wiley.com/doi/book/10.1002/9781118631980

    Hu, J., Fu, M.C. and Marcus, S.I., 2007.
    A model reference adaptive search method for global optimization.
    Operations Research, 55(3), pp.549-568.
    https://pubsonline.informs.org/doi/abs/10.1287/opre.1060.0367

    Kroese, D.P., Porotsky, S. and Rubinstein, R.Y., 2006.
    The cross-entropy method for continuous multi-extremal optimization.
    Methodology and Computing in Applied Probability, 8(3), pp.383-407.
    https://link.springer.com/article/10.1007/s11009-006-9753-0

    De Boer, P.T., Kroese, D.P., Mannor, S. and Rubinstein, R.Y., 2005.
    A tutorial on the cross-entropy method.
    Annals of Operations Research, 134(1), pp.19-67.
    https://link.springer.com/article/10.1007/s10479-005-5724-z

    Rubinstein, R.Y. and Kroese, D.P., 2004.
    The cross-entropy method: a unified approach to combinatorial optimization, Monte-Carlo simulation,
    and machine learning.
    New York: Springer.
    https://link.springer.com/book/10.1007/978-1-4757-4321-0
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # number of individuals/samples
            self.n_individuals = 1000
        assert self.n_individuals > 0
        if self.n_parents is None:  # number of elitists
            self.n_parents = 200
        assert self.n_parents > 0
        self.mean = options.get('mean')  # mean of Gaussian search (sampling/mutation) distribution
        if self.mean is None:
            self.mean = options.get('x')
        self.sigma = options.get('sigma')  # global (overall) step-size
        assert self.sigma is not None and self.sigma > 0.0
        self._sigmas = self.sigma*np.ones((self.ndim_problem,))  # individual step-sizes
        self._n_generations = 0
        self._printed_evaluations = self.n_function_evaluations

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

    def _collect(self, fitness, y=None, mean=None):
        self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results['mean'] = mean
        results['_sigmas'] = self._sigmas
        results['_n_generations'] = self._n_generations
        return results
