import numpy as np

from pypop7.optimizers.ds.ds import DS


class CS(DS):
    """Coordinate Search (CS).

    .. note:: `CS` is the *earliest* Direct (Pattern) Search method, at least dating back to Fermi (`The Nobel
       Prize in Physics 1938 <https://www.nobelprize.org/prizes/physics/1938/summary/>`_) and Metropolis (`IEEE Computer
       Society Computer Pioneer Award 1984 <https://en.wikipedia.org/wiki/Computer_Pioneer_Award>`_). Given that now
       it is *rarely* used to optimize black-box problems, it is **highly recommended** to first attempt other more
       advanced methods for large-scale black-box optimization (LSBBO).

       Its original version needs `3**n - 1` samples for each iteration in the worst case, where `n` is the
       dimensionality of the problem. Such a worst-case complexity limits its applicability for LSBBO severely.
       Instead, here we use the **opportunistic** strategy for simplicity. See Algorithm 3 from `Torczon, 1997, SIOPT
       <https://epubs.siam.org/doi/abs/10.1137/S1052623493250780>`_ for more details.

       AKA alternating directions, alternating variable search, axial relaxation, local variation, compass search.

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

                * 'gamma' - decreasing factor of global step-size (`float`, default: `0.5`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.ds.cs import CS
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3*numpy.ones((2,)),
       ...            'sigma': 1.0,
       ...            'verbose_frequency': 500}
       >>> cs = CS(problem, options)  # initialize the optimizer class
       >>> results = cs.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CS: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CS: 5000, 0.1491367032979898

    Attributes
    ----------
    gamma : `float`
            decreasing factor of global step-size.
    sigma : `float`
            final global step-size (changed during optimization).
    x     : `array_like`
            initial (starting) point.

    References
    ----------
    Larson, J., Menickelly, M. and Wild, S.M., 2019.
    Derivative-free optimization methods.
    Acta Numerica, 28, pp.287-404.
    https://tinyurl.com/4sr2t63j

    Audet, C. and Hare, W., 2017.
    Derivative-free and blackbox optimization.
    Berlin: Springer International Publishing.
    https://link.springer.com/book/10.1007/978-3-319-68913-5

    Torczon, V., 1997.
    On the convergence of pattern search algorithms.
    SIAM Journal on Optimization, 7(1), pp.1-25.
    https://epubs.siam.org/doi/abs/10.1137/S1052623493250780
    (See Algorithm 3 (Section 4.1) for details.)

    Fermi, E. and Metropolis N., 1952.
    Numerical solution of a minimum problem.
    Los Alamos Scientific Lab., Los Alamos, NM.
    https://www.osti.gov/servlets/purl/4377177
    """
    def __init__(self, problem, options):
        DS.__init__(self, problem, options)
        self.gamma = options.get('gamma', 0.5)  # decreasing factor of global step-size
        assert self.gamma > 0.0

    def initialize(self, args=None, is_restart=False):
        x = self._initialize_x(is_restart)  # initial point
        y = self._evaluate_fitness(x, args)  # fitness
        return x, y

    def iterate(self, x=None, args=None):
        improved, fitness = False, []
        for i in range(self.ndim_problem):  # to search along each coordinate
            for sgn in [-1, 1]:  # for two opponent directions
                if self._check_terminations():
                    return x, fitness
                xx = np.copy(x)
                xx[i] += sgn*self.sigma
                y = self._evaluate_fitness(xx, args)
                fitness.append(y)
                if y < self.best_so_far_y:
                    x = xx  # greedy / opportunistic
                    improved = True
                    break
        if not improved:  # to decrease step-size if no improvement
            self.sigma *= self.gamma
        return x, fitness

    def restart_reinitialize(self, args=None, x=None, y=None, fitness=None):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self._print_verbose_info(fitness, y)
            self.sigma = np.copy(self._sigma_bak)
            x, y = self.initialize(args, is_restart)
            self._fitness_list = [self.best_so_far_y]
            self._n_generations = 0
            self._n_restart += 1
            if self.verbose:
                print(' ....... *** restart *** .......')
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x, y = self.initialize(args)
        while True:
            self._print_verbose_info(fitness, y)
            x, y = self.iterate(x, args)
            if self._check_terminations():
                break
            self._n_generations += 1
            if self.is_restart:
                x, y = self.restart_reinitialize(args, x, y, fitness)
        return self._collect(fitness, y)
