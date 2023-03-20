import numpy as np
from scipy.optimize.optimize import _wrap_function as wf,\
    _line_for_search as ls, _minimize_scalar_bounded as msb

from pypop7.optimizers.ds.ds import DS


class POWELL(DS):
    """Powell's search method (POWELL).

     .. note:: This is a wrapper of the Powell algorithm from `SciPy
        <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html>`_ with accuracy control of
        maximum of function evaluations.

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
                * 'x' - initial (starting) point (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.ds.powell import POWELL
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3*numpy.ones((2,)),
       ...            'verbose_frequency': 500}
       >>> powell = POWELL(problem, options)  # initialize the optimizer class
       >>> results = powell.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"POWELL: {results['n_function_evaluations']}, {results['best_so_far_y']}")

    Attributes
    ----------
    x : `array_like`
        initial (starting) point.

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html

    Kochenderfer, M.J. and Wheeler, T.A., 2019.
    Algorithms for optimization.
    MIT Press.
    https://algorithmsbook.com/optimization/files/chapter-7.pdf
    (See Algorithm 7.3 (Page 102) for details.)

    Powell, M.J., 1964.
    An efficient method for finding the minimum of a function of several variables without calculating derivatives.
    The Computer Journal, 7(2), pp.155-162.
    https://academic.oup.com/comjnl/article-abstract/7/2/155/335330
    """
    def __init__(self, problem, options):
        DS.__init__(self, problem, options)
        self._func = None  # only for inner line searcher

    def initialize(self, args=None, is_restart=False):
        x = self._initialize_x(is_restart)  # initial (starting) search point
        y = self._evaluate_fitness(x, args)  # fitness
        u = np.identity(self.ndim_problem)
        if args is None:
            args = ()
        _, self._func = wf(self._evaluate_fitness, args=args)
        return x, y, u, y

    def _line_search(self, x, d, tol):
        def _func(alpha):  # only for line search
            return self._func(x + alpha*d)

        bound = ls(x, d, self.lower_boundary, self.upper_boundary)
        res = msb(_func, bound, xatol=tol/100.0)
        d *= res.x
        return res.fun, x + d, d

    def iterate(self, x=None, y=None, u=None, args=None):
        xx, yy = np.copy(x), np.copy(y)
        ind, delta, ys = 0, 0.0, []
        for i in range(self.ndim_problem):
            if self._check_terminations():
                return x, y, u, ys
            d, diff = u[i], y
            y, x, d = self._line_search(x, d, 1e-4*100)
            ys.append(y)
            diff -= y
            if diff > delta:
                delta, ind = diff, i
        d = x - xx  # extrapolated point
        _, ratio_e = ls(x, d, self.lower_boundary, self.upper_boundary)
        xxx = x + min(ratio_e, 1.0)*d
        yyy = self.fitness_function(xxx)
        if yy > yyy:
            t, temp = 2.0*(yy + yyy - 2.0*y), yy - y - delta
            t *= np.square(temp)
            temp = yy - yyy
            t -= delta*np.square(temp)
            if t < 0.0:
                y, x, d = self._line_search(x, d, 1e-4*100)
                ys.append(y)
                if np.any(d):
                    u[ind] = u[-1]
                    u[-1] = d
        self._n_generations += 1
        return x, y, u, ys

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x, y, u, yy = self.initialize(args)
        while not self.termination_signal:
            self._print_verbose_info(fitness, yy)
            x, y, u, yy = self.iterate(x, y, u, args)
        results = self._collect(fitness, yy)
        return results
