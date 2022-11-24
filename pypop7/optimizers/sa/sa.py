from pypop7.optimizers.rs import RS


class SA(RS):
    """Simulated Annealing (SA).

    This is the **base** (abstract) class for all SA classes. Please use any of its instantiated subclasses to
    optimize the black-box problem at hand.

    .. note:: `"Typical advantages of SA algorithms are their very mild memory requirements and the small
       computational effort per iteration."---[Bouttier&Gavra, 2019, JMLR]
       <https://www.jmlr.org/papers/v20/16-588.html>`_

       `"The SA algorithm can also be viewed as a local search algorithm in which there are occasional
       upward moves that lead to a cost increase. One hopes that such upward moves will help escape
       from local minima."---[Bertsimas&Tsitsiklis, 1993, Statistical Science]
       <https://tinyurl.com/yknunnpt>`_

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
              and with the following particular setting (`key`):
                * 'x' - initial (starting) point (`array_like`).

    Attributes
    ----------
    x     : `array_like`
            initial (starting) point.

    Methods
    -------

    References
    ----------
    Bouttier, C. and Gavra, I., 2019.
    Convergence rate of a simulated annealing algorithm with noisy observations.
    Journal of Machine Learning Research, 20(1), pp.127-171.
    https://www.jmlr.org/papers/v20/16-588.html

    Siarry, P., Berthiau, G., Durdin, F. and Haussy, J., 1997.
    Enhanced simulated annealing for globally minimizing functions of many-continuous variables.
    ACM Transactions on Mathematical Software, 23(2), pp.209-228.
    https://dl.acm.org/doi/abs/10.1145/264029.264043

    Bertsimas, D. and Tsitsiklis, J., 1993.
    Simulated annealing.
    Statistical Science, 8(1), pp.10-15.
    https://tinyurl.com/yknunnpt

    Corana, A., Marchesi, M., Martini, C. and Ridella, S., 1987.
    Minimizing multimodal functions of continuous variables with the "simulated annealing" algorithm.
    ACM Transactions on Mathematical Software, 13(3), pp.262-280.
    https://dl.acm.org/doi/abs/10.1145/29380.29864
    https://dl.acm.org/doi/10.1145/66888.356281

    Kirkpatrick, S., Gelatt, C.D. and Vecchi, M.P., 1983.
    Optimization by simulated annealing.
    Science, 220(4598), pp.671-680.
    https://science.sciencemag.org/content/220/4598/671

    Hastings, W.K., 1970.
    Monte Carlo sampling methods using Markov chains and their applications.
    Biometrika, 57(1), pp.97-109.
    https://academic.oup.com/biomet/article/57/1/97/284580

    Metropolis, N., Rosenbluth, A.W., Rosenbluth, M.N., Teller, A.H. and Teller, E., 1953.
    Equation of state calculations by fast computing machines.
    Journal of Chemical Physics, 21(6), pp.1087-1092.
    https://aip.scitation.org/doi/abs/10.1063/1.1699114
    """
    def __init__(self, problem, options):
        RS.__init__(self, problem, options)
        self.temperature = options.get('temperature')  # annealing temperature
        self.parent_x = None
        self.parent_y = None

    def initialize(self):
        raise NotImplementedError

    def iterate(self):  # for each iteration (generation)
        raise NotImplementedError
