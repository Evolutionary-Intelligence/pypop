import numpy as np
from scipy.optimize.optimize import Brent,\
    _wrap_function as wf, _line_for_search as ls, _minimize_scalar_bounded as msb

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
       POWELL: 5000, 0e0

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

    def _line_search(self, x, d, tol, y):
        def _func(alpha):
            return self._func(np.array(x + np.multiply(alpha, d)))

        if self.lower_boundary is None and self.upper_boundary is None:
            alpha_min, fret, _, _ = Brent(_func, full_output=1, tol=tol)
            d = alpha_min*x
            return np.squeeze(fret), x + d, d
        else:
            bound = ls(x, d, self.lower_boundary, self.upper_boundary)
            if np.isneginf(bound[0]) and np.isposinf(bound[1]):
                return self._line_search(x, d, tol, y)
            elif not np.isneginf(bound[0]) and not np.isposinf(bound[1]):
                res = msb(_func, bound, xatol=tol/100.0)
                d = res.x*d
                return np.squeeze(res.fun), x + d, d
            else:
                bound = np.arctan(bound[0]), np.arctan(bound[1])
                res = msb(lambda xx: _func(np.tan(xx)), bound, xatol=tol/100.0)
                d = np.tan(res.x)*d
                return np.squeeze(res.fun), x + d, d

    def iterate(self, x=None, y=None, u=None, args=None):
        xx, yy = np.copy(x), np.copy(y)
        ind, delta_k, ys = 0, 0, []
        for i in range(self.ndim_problem):
            if self._check_terminations():
                return x, y, u, ys
            d, diff = u[i], y
            y, x, d = self._line_search(x, d, 1e-4*100, y)
            ys.append(y)
            diff -= y
            if diff > delta_k:
                delta_k, ind = diff, i
        d = x - xx
        x1 = 2.0*x - xx
        fx1 = self.fitness_function(x1)
        if yy > fx1:
            t = 2.0*(yy + fx1 - 2.0*y)
            temp = yy - y - delta_k
            t *= temp**2
            temp = yy - fx1
            t -= delta_k*temp**2
            if t < 0.0:
                y, x, d = self._line_search(x, d, 1e-4*100, y)
                ys.append(y)
                if np.any(d):
                    u[ind] = u[-1]
                    u[-1] = d
        return x, y, u, ys

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x, y, u, yy = self.initialize(args)
        while not self.termination_signal:
            self._print_verbose_info(fitness, yy)
            x, y, u, yy = self.iterate(x, y, u, args)
            self._n_generations += 1
        results = self._collect(fitness, yy)
        return results
