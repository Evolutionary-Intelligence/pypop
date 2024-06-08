import numpy as np  # engine for numerical computing

from pypop7.optimizers.es.es import ES  # abstract class of all Evolution Strategies (ES) classes


class NES(ES):
    """Natural Evolution Strategies (NES).

    This is the **abstract** class for all `NES` classes. Please use any of its instantiated subclasses to optimize
    the **black-box** problem at hand.

    .. note:: `NES` is a family of **well-principled** population-based randomized search methods `with a relatively
       clean derivation from first principles <https://ieeexplore.ieee.org/abstract/document/4631255>`_, which
       maximize the expected fitness along with (estimated) `natural gradients
       <https://direct.mit.edu/neco/article-abstract/10/2/251/6143/Natural-Gradient-Works-Efficiently-in-Learning>`_.
       In this library, we have converted it to the *minimization* problem, in accordance with other modules.

    For some interesting applications of `NES`, please refer to `[Xu et al., 2024, ICLR]
    <https://openreview.net/pdf?id=6PbvbLyqT6>`_, `[Xuan Zhang et al., 2024, IEEE-LRA]
    <https://ieeexplore.ieee.org/document/10382561>`_, `[Conti et al., 2018, NeurIPS]
    <https://proceedings.neurips.cc/paper/2018/file/b1301141feffabac455e1f90a7de2054-Paper.pdf>`_, to name a few.

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
                    If not given, it will draw a random sample from the uniform distribution whose search
                    range is bounded by `problem['lower_boundary']` and `problem['upper_boundary']`, by
                    default.
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
    Hüttenrauch, M. and Neumann, G., 2024.
    `Robust black-box optimization for stochastic search and episodic reinforcement learning.
    <https://www.jmlr.org/papers/v25/22-0564.html>`_
    Journal of Machine Learning Research, 25(153), pp.1-44.
    
    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014.
    `Natural evolution strategies.
    <https://jmlr.org/papers/v15/wierstra14a.html>`_
    Journal of Machine Learning Research, 15(1), pp.949-980.

    Schaul, T., 2011.
    `Studies in continuous black-box optimization.
    <https://people.idsia.ch/~schaul/publications/thesis.pdf>`_
    Doctoral Dissertation, Technische Universität München.

    Yi, S., Wierstra, D., Schaul, T. and Schmidhuber, J., 2009, June.
    `Stochastic search using the natural gradient.
    <https://doi.org/10.1145/1553374.1553522>`_
    In Proceedings of International Conference on Machine Learning (pp. 1161-1168).

    Wierstra, D., Schaul, T., Peters, J. and Schmidhuber, J., 2008, June.
    `Natural evolution strategies.
    <https://doi.org/10.1109/CEC.2008.4631255>`_
    In IEEE Congress on Evolutionary Computation (pp. 3381-3387). IEEE.

    Please refer to the *official* Python source code from `PyBrain` (now not actively maintained):
    https://github.com/pybrain/pybrain
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self._u = None  # for fitness shaping

    def initialize(self):
        r, _u = np.arange(self.n_individuals), np.zeros((self.n_individuals,))
        for i in range(self.n_individuals):
            if r[i] >= self.n_individuals*0.5:
                _u[i] = r[i] - self.n_individuals*0.5
        self._u = _u/np.max(_u)

    def iterate(self):
        raise NotImplementedError
