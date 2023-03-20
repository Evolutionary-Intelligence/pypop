import numpy as np
from scipy.optimize.optimize import _wrap_function as wf,\
    _line_for_search as ls, _status_message as sm, OptimizeResult as OR, bracket as bk

from pypop7.optimizers.ds.ds import DS


class Brent:
    #  need to rethink design of __init__
    def __init__(self, func, args=(), tol=1.48e-8, maxiter=500):
        self.func = func
        self.args = args
        self.tol = tol
        self.maxiter = maxiter
        self._mintol = 1.0e-11
        self._cg = 0.3819660
        self.xmin = None
        self.fval = None
        self.iter = 0
        self.funcalls = 0
        self.fitness = []

    # need to rethink design of set_bracket (new options, etc.)
    def set_bracket(self, brack=None):
        self.brack = brack

    def get_bracket_info(self):
        # set up
        func = self.func
        args = self.args
        brack = self.brack
        # BEGIN core bracket_info code #
        # carefully DOCUMENT any CHANGES in core #
        if brack is None:
            xa, xb, xc, fa, fb, fc, funcalls = bk(func, args=args)
        elif len(brack) == 2:
            xa, xb, xc, fa, fb, fc, funcalls = bk(func, xa=brack[0], xb=brack[1], args=args)
        elif len(brack) == 3:
            xa, xb, xc = brack
            if xa > xc:  # swap so xa < xc can be assumed
                xc, xa = xa, xc
            if not ((xa < xb) and (xb < xc)):
                raise ValueError("Not a bracketing interval.")
            fa = func(*((xa,) + args))
            fb = func(*((xb,) + args))
            fc = func(*((xc,) + args))
            if not ((fb < fa) and (fb < fc)):
                raise ValueError("Not a bracketing interval.")
            funcalls = 3
        else:
            raise ValueError("Bracketing interval must be "
                             "length 2 or 3 sequence.")
        # END core bracket_info code #

        return xa, xb, xc, fa, fb, fc, funcalls

    def optimize(self):
        # set up for optimization
        func = self.func
        xa, xb, xc, fa, fb, fc, funcalls = self.get_bracket_info()
        self.fitness.append(fa)
        self.fitness.append(fb)
        self.fitness.append(fc)
        _mintol = self._mintol
        _cg = self._cg
        #################################
        # BEGIN CORE ALGORITHM
        #################################
        x = w = v = xb
        fw = fv = fx = func(*((x,) + self.args))
        self.fitness.append(fx)
        if xa < xc:
            a = xa
            b = xc
        else:
            a = xc
            b = xa
        deltax = 0.0
        funcalls += 1
        iter = 0
        while iter < self.maxiter:
            tol1 = self.tol * np.abs(x) + _mintol
            tol2 = 2.0 * tol1
            xmid = 0.5 * (a + b)
            # check for convergence
            if np.abs(x - xmid) < (tol2 - 0.5 * (b - a)):
                break
            # XXX In the first iteration, rat is only bound in the true case
            # of this conditional. This used to cause an UnboundLocalError
            # (gh-4140). It should be set before the if (but to what?).
            if np.abs(deltax) <= tol1:
                if x >= xmid:
                    deltax = a - x       # do a golden section step
                else:
                    deltax = b - x
                rat = _cg * deltax
            else:                              # do a parabolic step
                tmp1 = (x - w) * (fx - fv)
                tmp2 = (x - v) * (fx - fw)
                p = (x - v) * tmp2 - (x - w) * tmp1
                tmp2 = 2.0 * (tmp2 - tmp1)
                if tmp2 > 0.0:
                    p = -p
                tmp2 = np.abs(tmp2)
                dx_temp = deltax
                deltax = rat
                # check parabolic fit
                if ((p > tmp2 * (a - x)) and (p < tmp2 * (b - x)) and
                        (np.abs(p) < np.abs(0.5 * tmp2 * dx_temp))):
                    rat = p * 1.0 / tmp2        # if parabolic step is useful.
                    u = x + rat
                    if (u - a) < tol2 or (b - u) < tol2:
                        if xmid - x >= 0:
                            rat = tol1
                        else:
                            rat = -tol1
                else:
                    if x >= xmid:
                        deltax = a - x  # if it's not do a golden section step
                    else:
                        deltax = b - x
                    rat = _cg * deltax

            if np.abs(rat) < tol1:            # update by at least tol1
                if rat >= 0:
                    u = x + tol1
                else:
                    u = x - tol1
            else:
                u = x + rat
            fu = func(*((u,) + self.args))      # calculate new output value
            self.fitness.append(fu)
            funcalls += 1

            if fu > fx:                 # if it's bigger than current
                if u < x:
                    a = u
                else:
                    b = u
                if (fu <= fw) or (w == x):
                    v = w
                    w = u
                    fv = fw
                    fw = fu
                elif (fu <= fv) or (v == x) or (v == w):
                    v = u
                    fv = fu
            else:
                if u >= x:
                    a = x
                else:
                    b = x
                v = w
                w = x
                x = u
                fv = fw
                fw = fx
                fx = fu

            iter += 1
        #################################
        # END CORE ALGORITHM
        #################################

        self.xmin = x
        self.fval = fx
        self.iter = iter
        self.funcalls = funcalls

    def get_result(self, full_output=False):
        if full_output:
            return self.xmin, self.fval, self.iter, self.funcalls, self.fitness
        else:
            return self.xmin


def brent(func, args=(), brack=None, tol=1.48e-8, full_output=0, maxiter=500):
    options = {'xtol': tol,
               'maxiter': maxiter}
    res = _minimize_scalar_brent(func, brack, args, **options)
    if full_output:
        return res['x'], res['fun'], res['nit'], res['nfev'], res['fitness']
    else:
        return res['x']


def _minimize_scalar_brent(func, brack=None, args=(),
                           xtol=1.48e-8, maxiter=500):
    tol = xtol
    if tol < 0:
        raise ValueError('tolerance should be >= 0, got %r' % tol)

    tbrent = Brent(func=func, args=args, tol=tol, full_output=True, maxiter=maxiter)
    tbrent.set_bracket(brack)
    tbrent.optimize()
    x, fval, nit, nfev, fitness = tbrent.get_result(full_output=True)

    success = nit < maxiter and not (np.isnan(x) or np.isnan(fval))

    return OR(fun=fval, x=x, nit=nit, nfev=nfev, success=success, fitness=fitness)


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
                * 'sigma' - initial global step-size (`float`, default: `1.0`),
                * 'x'     - initial (starting) point (`array_like`),

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
       ...            'sigma': 0.1,
       ...            'verbose_frequency': 500}
       >>> powell = POWELL(problem, options)  # initialize the optimizer class
       >>> results = powell.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"POWELL: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       POWELL: 5000, 0e0

    Attributes
    ----------
    sigma : `float`
            initial global step-size.
    x     : `array_like`
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
        self._func = None

    def initialize(self, args=None, is_restart=False):
        x = self._initialize_x(is_restart)  # initial (starting) search point
        y = self._evaluate_fitness(x, args)  # fitness
        u = np.identity(self.ndim_problem)
        if args is None:
            args = ()
        _, self._func = wf(self._evaluate_fitness, args=args)
        return x, y, u, y

    def _minimize_scalar_bounded(self, func, bounds, args=(), xatol=1e-5, maxiter=500):
        maxfun = maxiter
        # Test bounds are of correct form
        if len(bounds) != 2:
            raise ValueError('bounds must have two elements.')
        x1, x2 = bounds

        if not (np.isscalar(x1) and np.isscalar(x2)):
            raise ValueError("Optimization bounds must be scalars"
                             " or array scalars.")
        if x1 > x2:
            raise ValueError("The lower bound exceeds the upper bound.")

        flag = 0

        sqrt_eps = np.sqrt(2.2e-16)
        golden_mean = 0.5 * (3.0 - np.sqrt(5.0))
        a, b = x1, x2
        fulc = a + golden_mean * (b - a)
        nfc, xf = fulc, fulc
        rat = e = 0.0
        x = xf
        fun = []
        fx = func(x, *args)
        fun.append(fx)
        num = 1
        fu = np.inf

        ffulc = fnfc = fx
        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        while np.abs(xf - xm) > (tol2 - 0.5 * (b - a)):
            golden = 1
            # Check for parabolic fit
            if np.abs(e) > tol1:
                golden = 0
                r = (xf - nfc) * (fx - ffulc)
                q = (xf - fulc) * (fx - fnfc)
                p = (xf - fulc) * q - (xf - nfc) * r
                q = 2.0 * (q - r)
                if q > 0.0:
                    p = -p
                q = np.abs(q)
                r = e
                e = rat

                # Check for acceptability of parabola
                if ((np.abs(p) < np.abs(0.5 * q * r)) and (p > q * (a - xf)) and
                        (p < q * (b - xf))):
                    rat = (p + 0.0) / q
                    x = xf + rat

                    if ((x - a) < tol2) or ((b - x) < tol2):
                        si = np.sign(xm - xf) + ((xm - xf) == 0)
                        rat = tol1 * si
                else:  # do a golden-section step
                    golden = 1

            if golden:  # do a golden-section step
                if xf >= xm:
                    e = a - xf
                else:
                    e = b - xf
                rat = golden_mean * e

            si = np.sign(rat) + (rat == 0)
            x = xf + si * np.maximum(np.abs(rat), tol1)
            fu = func(x, *args)
            num += 1
            fun.append(fu)

            if fu <= fx:
                if x >= xf:
                    a = xf
                else:
                    b = xf
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = xf, fx
                xf, fx = x, fu
            else:
                if x < xf:
                    a = x
                else:
                    b = x
                if (fu <= fnfc) or (nfc == xf):
                    fulc, ffulc = nfc, fnfc
                    nfc, fnfc = x, fu
                elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                    fulc, ffulc = x, fu

            xm = 0.5 * (a + b)
            tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
            tol2 = 2.0 * tol1

            if num >= maxfun:
                flag = 1
                break

        if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
            flag = 2

        fval = fx

        result = OR(fun=fval, status=flag, success=(flag == 0),
                    message={0: 'Solution found.',
                             1: 'Maximum number of function calls ''reached.',
                             2: sm['nan']}.get(flag, ''),
                    x=xf, nfev=num, fitness=fun)

        return result

    def _line_search(self, x, d, tol, y):
        def _func(alpha):
            return self._func(np.array(x + np.multiply(alpha, d)))

        if self.lower_boundary is None and self.upper_boundary is None:
            alpha_min, fret, _, _, fitness = brent(_func, full_output=1, tol=tol)
            d = alpha_min*x
            return np.squeeze(fret), x + d, d, fitness
        else:
            bound = ls(x, d, self.lower_boundary, self.upper_boundary)
            if np.isneginf(bound[0]) and np.isposinf(bound[1]):
                return self._line_search(x, d, tol, y)
            elif not np.isneginf(bound[0]) and not np.isposinf(bound[1]):
                res = self._minimize_scalar_bounded(_func, bound, xatol=tol/100.0)
                d = res.x*d
                return np.squeeze(res.fun), x + d, d, res.fitness
            else:
                bound = np.arctan(bound[0]), np.arctan(bound[1])
                res = self._minimize_scalar_bounded(lambda xx: _func(np.tan(xx)), bound, xatol=tol/100.0)
                d = np.tan(res.x)*d
                return np.squeeze(res.fun), x + d, d, res.fitness

    def iterate(self, x=None, y=None, u=None, args=None):
        xx, yy = np.copy(x), np.copy(y)
        ind, delta_k, ys = 0, 0, []
        for i in range(self.ndim_problem):
            if self._check_terminations():
                return x, y, u, ys
            d, diff = u[i], y
            y, x, d, yyy = self._line_search(x, d, 1e-4*100, y)
            ys.extend(yyy)
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
                y, x, d, yyy = self._line_search(x, d, 1e-4*100, y)
                ys.extend(yyy)
                if np.any(d):
                    u[ind] = u[-1]
                    u[-1] = d
        return x, y, u, ys

    def _check_success(self):
        if (self.upper_boundary is not None) and (self.lower_boundary is not None) and (
                np.any(self.lower_boundary > self.best_so_far_x) or np.any(self.best_so_far_x > self.upper_boundary)):
            return False
        elif np.isnan(self.best_so_far_y) or np.any(np.isnan(self.best_so_far_x)):
            return False
        return True

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x, y, u, yy = self.initialize(args)
        while not self.termination_signal:
            self._print_verbose_info(fitness, yy)
            x, y, u, yy = self.iterate(x, y, u, args)
            self._n_generations += 1
        yy.append(y)
        results = self._collect(fitness, yy)
        results['success'] = self._check_success()
        return results
