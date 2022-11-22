from pypop7.optimizers.rs import RS


class SA(RS):
    """Simulated Annealing (SA).

    This is the **base** (abstract) class for all SA classes. Please use any of its instantiated subclasses to
    optimize the black-box problem at hand.

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
    Siarry, P., Berthiau, G., Durdin, F. and Haussy, J., 1997.
    Enhanced simulated annealing for globally minimizing functions of many-continuous variables.
    ACM Transactions on Mathematical Software, 23(2), pp.209-228.
    https://dl.acm.org/doi/abs/10.1145/264029.264043

    Corana, A., Marchesi, M., Martini, C. and Ridella, S., 1987.
    Minimizing multimodal functions of continuous variables with the "simulated annealing" algorithm.
    ACM Transactions on Mathematical Software, 13(3), pp.262-280.
    https://dl.acm.org/doi/abs/10.1145/29380.29864
    https://dl.acm.org/doi/10.1145/66888.356281

    Kirkpatrick, S., Gelatt, C.D. and Vecchi, M.P., 1983.
    Optimization by simulated annealing.
    Science, 220(4598), pp.671-680.
    https://science.sciencemag.org/content/220/4598/671
    """
    def __init__(self, problem, options):
        RS.__init__(self, problem, options)

    def initialize(self):
        raise NotImplementedError

    def iterate(self):  # for each iteration (generation)
        raise NotImplementedError
