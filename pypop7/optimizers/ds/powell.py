import numpy as np
import time

from pypop7.optimizers.ds.ds import DS
from scipy.optimize.optimize import Brent
import scipy.optimize.optimize as Optimize


class POWELL(DS):
    """Powell's Method(POWELL).

     .. note:: `"The algorithm is adapted from the powell algorithm created by scipy.
       https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html

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
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'x'             - initial (starting) point (`array_like`),
                * 'sigma'         - initial (global) step-size (`float`),
                * 'xtol'          - factor for linear search (`float`, default: `1e-4`).

    Examples
    --------
    Use the Direct Search optimizer `POWELL` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.ds.powell import POWELL
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3 * numpy.ones((2,)),
       ...            'sigma': 0.1,
       ...            'verbose_frequency': 500}
       >>> powell = POWELL(problem, options)  # initialize the optimizer class
       >>> results = powell.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"POWELL: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       POWELL: 5000, 0e0

    Attributes
    ----------
    x              : `array_like`
                      starting search point.
    sigma          : `float`
                      final (global) step-size.
    xtol           : `float`
                      Relative error in solution `xopt` acceptable for convergence.

    Reference
    ---------
    Kochenderfer, M.J. and Wheeler, T.A., 2019.
    Algorithms for optimization.
    MIT Press.
    https://algorithmsbook.com/optimization/files/chapter-7.pdf
    (See Algorithm 7.3 (Page 102) for details.)

    M. J. D. Powell
    An Efficient Method for Finding the Minimum of a Function
    of Several Variables Without Calculating Derivative
    Comput.J. 7(2)155-162(1964)
    https://academic.oup.com/comjnl/article-abstract/7/2/155/335330?redirectedFrom=fulltext&login=false

    """
    def __init__(self, problem, options, args=None):
        DS.__init__(self, problem, options)
        self.xtol = options.get("xtol", 1e-4)
        fcalls, self.func = Optimize._wrap_function(self.fitness_function, args=())

    def initialize(self, args=None, is_restart=False):
        x = self._initialize_x(is_restart)  # initial point
        y = self._evaluate_fitness(x, args)  # fitness
        u = np.identity(self.ndim_problem)
        return x, y, u

    def line_search(self, x0=None, d=None, tol=1e-3, lower_bound=None, upper_bound=None, y=None, args=None):
        def myfunc(alpha):
            return self.func(np.array(x0 + np.multiply(alpha, d)))

        if lower_bound is None and upper_bound is None:
            alpha_min, fret, _, _ = Brent(myfunc, full_output=1, tol=tol)
            d = alpha_min * x0
            return np.squeeze(fret), x0+d, d
        else:
            bound = Optimize._line_for_search(x0, d, lower_bound, upper_bound)
            if np.isneginf(bound[0]) and np.isposinf(bound[1]):
                return self.line_search(x0, d, y=y, tol=tol)
            elif not np.isneginf(bound[0]) and not np.isposinf(bound[1]):
                res = Optimize._minimize_scalar_bounded(myfunc, bound, xatol=tol/100)
                d = res.x * d
                return np.squeeze(res.fun), x0 + d, d
            else:
                bound = np.arctan(bound[0]), np.arctan(bound[1])
                res = Optimize._minimize_scalar_bounded(lambda x: myfunc(np.tan(x)),
                                                        bound,
                                                        xatol=tol/100)
                d = np.tan(res.x) * d
                return np.squeeze(res.fun), x0 + d, d

    def iterate(self, x=None, y=None, u=None, args=None):
        xx, yy = np.copy(x), np.copy(y)
        ind, delta_k = 0, 0
        ys = []
        for i in range(self.ndim_problem):
            if self._check_terminations():
                return x, y, u, ys
            d = u[i]
            diff = y
            y, x, d = self.line_search(x, d, tol=self.xtol*100,
                                       lower_bound=self.lower_boundary,
                                       upper_bound=self.upper_boundary,
                                       y=y)
            ys.append(y)
            if y < self.best_so_far_y:
                self.best_so_far_x, self.best_so_far_y = np.copy(x), np.copy(y)
            self.time_function_evaluations += time.time() - self.start_function_evaluations
            self.n_function_evaluations += 1
            diff -= y
            if diff > delta_k:
                delta_k = diff
                ind = i
        d = x - xx
        x1 = 2 * x - xx
        fx1 = self.fitness_function(x1)
        if yy > fx1:
            t = 2.0 * (yy + fx1 - 2.0 * y)
            temp = (yy - y - delta_k)
            t *= temp ** 2
            temp = yy - fx1
            t -= delta_k * temp ** 2
            if t < 0.0:
                y, x, d = self.line_search(x, d, tol=self.xtol * 100,
                                           lower_bound=self.lower_boundary,
                                           upper_bound=self.upper_boundary,
                                           y=y)
                ys.append(y)
                if y < self.best_so_far_y:
                    self.best_so_far_x, self.best_so_far_y = np.copy(x), np.copy(y)
                self.time_function_evaluations += time.time() - self.start_function_evaluations
                self.n_function_evaluations += 1
                if np.any(d):
                    u[ind] = u[-1]
                    u[-1] = d
        return x, y, u, ys

    def _check_success(self):
        if self.upper_boundary and (np.any(self.lower_boundary > self.best_so_far_x)
                                    or np.any(self.best_so_far_x > self.upper_boundary)):
            return False
        elif np.isnan(self.best_so_far_y) or np.isnan(self.best_so_far_x).any():
            return False
        return True

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x, y, u = self.initialize(args)
        ys = y
        while not self.termination_signal:
            self._print_verbose_info(fitness, ys)
            x, y, u, ys = self.iterate(x, y, u, args)
            self._n_generations += 1
        results = self._collect(fitness, y)
        results["success"] = self.check_success()
        return results
