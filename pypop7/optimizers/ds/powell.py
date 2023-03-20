import numpy as np
from scipy.optimize.optimize import _wrap_function as wf, _line_for_search as ls

from pypop7.optimizers.ds.ds import DS


def _minimize_scalar_bounded(func, bounds,
                             max_function_evaluations, fitness_threshold,
                             tol=1e-5, max_iterations=500):
    # this is adopted from https://github.com/scipy/scipy/blob/main/scipy/optimize/_optimize.py
    #   with slight modifications
    n_function_evaluations, num_iterations, yy = 0, 0, []
    a, b = bounds
    sqrt_eps, golden_mean = 1.4832396974191326e-08, 0.3819660112501051
    gm = a + golden_mean*(b - a)
    gm_1 = gm_2 = gm
    rat = e = 0.0
    y = func(gm_2)
    n_function_evaluations += 1
    yy.append(y)
    if (n_function_evaluations == max_function_evaluations) or (y < fitness_threshold):
        return y, gm_2, yy
    y_1 = y_2 = y
    middle = 0.5*(a + b)
    tol_1 = sqrt_eps*np.abs(gm_2) + tol/3.0
    tol_2 = 2.0*tol_1
    while np.abs(gm_2 - middle) > (tol_2 - 0.5*(b - a)):
        golden = 1
        if np.abs(e) > tol_1:
            golden = 0
            r = (gm_2 - gm_1)*(y - y_1)
            q = (gm_2 - gm)*(y - y_2)
            p = (gm_2 - gm)*q - (gm_2 - gm_1)*r
            q = 2.0*(q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            r = e
            e = rat
            if (np.abs(p) < np.abs(0.5*q*r)) and (p > q*(a - gm_2)) and (p < q*(b - gm_2)):
                rat = (p + 0.0)/q
                x = gm_2 + rat
                if ((x - a) < tol_2) or ((b - x) < tol_2):
                    rat = tol_1*(np.sign(middle - gm_2) + ((middle - gm_2) == 0))
            else:
                golden = 1
        if golden:
            if gm_2 >= middle:
                e = a - gm_2
            else:
                e = b - gm_2
            rat = golden_mean*e
        x = gm_2 + (np.sign(rat) + (rat == 0))*np.maximum(np.abs(rat), tol_1)
        yyy = func(x)
        n_function_evaluations += 1
        yy.append(y)
        if (n_function_evaluations == max_function_evaluations) or (y < fitness_threshold):
            return y, gm_2, yy
        if yyy <= y:
            if x >= gm_2:
                a = gm_2
            else:
                b = gm_2
            gm, y_1 = gm_1, y_2
            gm_1, y_2 = gm_2, y
            gm_2, y = x, yyy
        else:
            if x < gm_2:
                a = x
            else:
                b = x
            if (yyy <= y_2) or (gm_1 == gm_2):
                gm, y_1 = gm_1, y_2
                gm_1, y_2 = x, yyy
            elif (yyy <= y_1) or (gm == gm_2) or (gm == gm_1):
                gm, y_1 = x, yyy
        middle = 0.5*(a + b)
        tol_1 = sqrt_eps*np.abs(gm_2) + tol/3.0
        tol_2 = 2.0*tol_1
        num_iterations += 1
        if num_iterations == max_iterations - 1:
            break
    return y, gm_2, yy


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
       ...            'ndim_problem': 20,
       ...            'lower_boundary': -5*numpy.ones((20,)),
       ...            'upper_boundary': 5*numpy.ones((20,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3*numpy.ones((20,)),
       ...            'verbose_frequency': 500}
       >>> powell = POWELL(problem, options)  # initialize the optimizer class
       >>> results = powell.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"POWELL: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       POWELL: 50000, 0.0

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

    def _line_search(self, x, d, tol=1e-4*100):
        def _func(alpha):  # only for line search
            return self._func(x + alpha*d)

        bound = ls(x, d, self.lower_boundary, self.upper_boundary)
        y, gm, yy = _minimize_scalar_bounded(_func, bound,
                                             self.max_function_evaluations - self.n_function_evaluations,
                                             self.fitness_threshold, tol/100.0)
        d *= gm
        return y, x + d, d, yy

    def iterate(self, x=None, y=None, u=None, args=None):
        xx, yy = np.copy(x), np.copy(y)
        big_ind, delta, ys = 0, 0.0, []
        for i in range(self.ndim_problem):
            if self._check_terminations():
                return x, y, u, ys
            d, diff = u[i], y
            y, x, d, fitness = self._line_search(x, d)
            ys.extend(fitness)
            diff -= y
            if diff > delta:
                delta, big_ind = diff, i
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
                y, x, d, fitness = self._line_search(x, d)
                ys.extend(fitness)
                if np.any(d):
                    u[big_ind] = u[-1]
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
