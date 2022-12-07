import numpy as np

from pypop7.optimizers.ds.ds import DS


class POWELL(DS):
    """Powell's Method(POWELL).
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
                * 'eps'           - factor for linear search (`float`, default: `1e-6`).
                * 'initial_step'  - factor for generate brackets (`float`, default: `1e-6`).
                * 'grow_limit'    - factor for generate brackets (`float`, default: `100.0`).

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
       POWELL: 248, 5.406485438404457e-14

    Attributes
    ----------
    x              : `array_like`
                      starting search point.
    sigma          : `float`
                      final (global) step-size.
    eps            : `float`
                      factor for linear search
    initial_step   : `float`
                      factor for generate brackets
    grow_limit     : `float`
                      factor for generate brackets

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
    def __init__(self, problem, options):
        DS.__init__(self, problem, options)
        self.eps = options.get("eps", 1e-6)
        self.initial_step = options.get("initial_step", 1e-6)
        self.grow_limit = options.get("grow_limit", 100.0)

    def initialize(self, args=None, is_restart=False):
        x = self._initialize_x(is_restart)  # initial point
        y = self._evaluate_fitness(x, args)  # fitness
        u = np.identity(self.ndim_problem)
        return x, y, u

    def initialize_brackets(self, x0, y, d):
        gold = (np.sqrt(5) + 1) / 2
        t = 1e-20
        ax, bx = np.zeros((self.ndim_problem,)), self.initial_step * np.ones((self.ndim_problem,))
        fa, fb = y, self._evaluate_fitness(x0 + bx * d)
        if fb > fa:
            ax, bx = bx, ax
            fa, fb = fb, fa
        cx = bx + gold * (bx - ax)
        fc = self._evaluate_fitness(x0 + cx * d)
        while fb > fc:
            r = (bx - ax) * (fb - fc)
            q = (bx - cx) * (fb - fa)
            if min(abs(q - r)) < t:
                s = np.sign(q - r) * t
            else:
                s = q - r
            u = bx - ((bx - cx) * q - (bx - ax) * r) / (2 * s)
            ulim = bx + self.grow_limit * (cx - bx)
            if np.dot((bx - u), (u - cx)) > 0.0:
                fu = self._evaluate_fitness(x0 + u * d)
                if fu < fc:
                    ax = bx
                    bx = u
                    fa = fb
                    fb = fu
                    break
                elif fu > fb:
                    cx = u
                    fc = fu
                    break
                u = cx + gold * (cx - bx)
                fu = self._evaluate_fitness(x0 + u * d)
            elif np.dot((cx - u), (u - ulim)) > 0.0:
                fu = self._evaluate_fitness(x0 + u * d)
                if fu < fc:
                    bx, cx, u = cx, u, u + gold * (u - cx)
                    fb, fc, fu = fc, fu, self._evaluate_fitness(x0 + u * d)
            elif np.dot((u - ulim), (ulim - cx)) >= 0.0:
                u = ulim
                fu = self._evaluate_fitness(x0 + u * d)
            else:
                u = cx + gold * (cx - bx)
                fu = self._evaluate_fitness(x0 + u * d)
            ax, bx, cx = bx, cx, u
            fa, fb, fc = fb, fc, fu
        return sorted([(fa, ax), (fb, bx), (fc, cx)], key=lambda x: x[0])

    def line_search(self, x0=None, y=None, d=None, args=None):
        u, fu = np.inf, np.inf
        (fa, a), (fb, b), (fc, c) = self.initialize_brackets(x0, y, d)
        num = (b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)
        den = 2 * ((b - a) * (fb - fc) - (b - c) * (fb - fa))
        while max(abs(u - (b - num/den))) > self.eps:
            u = b - num / den
            fu = self._evaluate_fitness(x0 + u * d, args)
            points = sorted([(fa, a), (fb, b), (fc, c), (fu, u)], key=lambda x: x[0])[:-1]
            (fa, a), (fb, b), (fc, c) = sorted(points, key=lambda x: x[0])
            num = (b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)
            den = 2 * ((b - a) * (fb - fc) - (b - c) * (fb - fa))
        return sorted([(fa, a), (fb, b), (fc, c), (fu, u)], key=lambda x: x[0])[0][::-1]

    def iterate(self, args=None, x=None, y=None, u=None):
        xx, yy = np.copy(x), np.copy(y)
        ind, delta_k = 0, 0
        for i in range(self.ndim_problem):
            if self._check_terminations():
                return x, y, u
            d = u[i]
            diff = y
            alpha, y = self.line_search(x, y, d)
            x += alpha * d
            diff -= y
            if diff > delta_k:
                delta_k = diff
                ind = i
        d = x - xx
        x1 = 2 * x - xx
        fx1 = self._evaluate_fitness(x1)
        if yy > fx1:
            t = 2.0 * (yy + fx1 - 2.0 * y)
            temp = (yy - y - delta_k)
            t *= temp ** 2
            temp = yy - fx1
            t -= delta_k * temp ** 2
            if t < 0.0:
                alpha, y = self.line_search(x, y, d)
                x += alpha * d
                u[ind] = u[-1]
                u[-1] = d
        return x, y, u

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x, y, u = self.initialize(args)
        fitness.append(y)
        while True:
            x, y, u = self.iterate(args, x, y, u)
            if self.saving_fitness:
                fitness.append(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results
