from pypop7.optimizers.rs import RS  # abstract class for all Random Search (RS) subclasses


class SA(RS):
    """Simulated Annealing (SA).

    This is the **abstract** class for all Simulated Annealing (`SA`) classes. Please use any of its
    instantiated subclasses to optimize the black-box problem at hand.

    .. note:: `"Typical advantages of SA algorithms are their very mild memory requirements and the
       small computational effort per iteration."---[Bouttier&Gavra, 2019, JMLR]
       <https://www.jmlr.org/papers/v20/16-588.html>`_

       `"The SA algorithm can also be viewed as a local search algorithm in which there are occasional
       upward moves that lead to a cost increase. One hopes that such upward moves will help escape
       from local minima."---[Bertsimas&Tsitsiklis, 1993, Statistical Science]
       <https://doi.org/10.1214/ss/1177011077>`_

    For its `pytest <https://docs.pytest.org/>`_ based testing, please refer to `this Python code
    <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/sa/test_sa.py>`_.

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
                * 'temperature' - annealing temperature (`float`),
                * 'x'           - initial (starting) point (`array_like`).

    Attributes
    ----------
    temperature : `float`
                  annealing temperature.
    x           : `array_like`
                  initial (starting) point.

    Methods
    -------

    References
    ----------
    Bras, P., 2024.
    `Convergence of Langevin-simulated annealing algorithms with multiplicative noise.
    <https://www.degruyter.com/document/doi/10.1515/mcma-2023-2009/pdf>`_
    Mathematics of Computation, 93(348), pp.1761-1803.

    Bouttier, C. and Gavra, I., 2019.
    `Convergence rate of a simulated annealing algorithm with noisy observations.
    <https://www.jmlr.org/papers/v20/16-588.html>`_
    Journal of Machine Learning Research, 20(1), pp.127-171.

    Lecchini-Visintini, A., Lygeros, J. and Maciejowski, J., 2007.
    `Simulated annealing: Rigorous finite-time guarantees for optimization on continuous domains.
    <https://tinyurl.com/333m2fnu>`_
    Advances in Neural Information Processing Systems, 20.

    Siarry, P., Berthiau, G., Durdin, F. and Haussy, J., 1997.
    `Enhanced simulated annealing for globally minimizing functions of many-continuous variables.
    <https://dl.acm.org/doi/abs/10.1145/264029.264043>`_
    ACM Transactions on Mathematical Software, 23(2), pp.209-228.

    Granville, V., Krivánek, M. and Rasson, J.P., 1994.
    Simulated annealing: A proof of convergence.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 16(6), pp.652-656.

    Bertsimas, D. and Tsitsiklis, J., 1993.
    `Simulated annealing.
    <https://doi.org/10.1214/ss/1177011077>`_
    Statistical Science, 8(1), pp.10-15.

    Corana, A., Marchesi, M., Martini, C. and Ridella, S., 1987.
    `Minimizing multimodal functions of continuous variables with the "simulated annealing" algorithm.
    <https://dl.acm.org/doi/abs/10.1145/29380.29864>`_
    ACM Transactions on Mathematical Software, 13(3), pp.262-280.
    https://dl.acm.org/doi/10.1145/66888.356281

    Szu, H.H. and Hartley, R.L., 1987.
    `Nonconvex optimization by fast simulated annealing.
    <https://ieeexplore.ieee.org/abstract/document/1458183>`_
    Proceedings of the IEEE, 75(11), pp.1538-1540.

    Kirkpatrick, S., Gelatt, C.D. and Vecchi, M.P., 1983.
    `Optimization by simulated annealing.
    <https://science.sciencemag.org/content/220/4598/671>`_
    Science, 220(4598), pp.671-680.

    Hastings, W.K., 1970.
    `Monte Carlo sampling methods using Markov chains and their applications.
    <https://academic.oup.com/biomet/article/57/1/97/284580>`_
    Biometrika, 57(1), pp.97-109.

    Metropolis, N., Rosenbluth, A.W., Rosenbluth, M.N., Teller, A.H. and Teller, E., 1953.
    `Equation of state calculations by fast computing machines.
    <https://aip.scitation.org/doi/abs/10.1063/1.1699114>`_
    Journal of Chemical Physics, 21(6), pp.1087-1092.
    """
    def __init__(self, problem, options):
        RS.__init__(self, problem, options)
        self.temperature = options.get('temperature')  # annealing temperature
        self.parent_x, self.parent_y = None, None

    def initialize(self):
        raise NotImplementedError

    def iterate(self):  # for each iteration (generation)
        raise NotImplementedError
