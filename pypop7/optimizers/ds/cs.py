import numpy as np

from pypop7.optimizers.ds.ds import DS


class CS(DS):
    """Coordinate Search (CS).

    .. note:: `CS` is one of the *earliest* Direct (Pattern) Search methods, at least dating back to Fermi (`The Nobel
       Prize in Physics 1938 <https://www.nobelprize.org/prizes/physics/1938/summary/>`_) and Metropolis (`IEEE Computer
       Society Computer Pioneer Award 1984 <https://en.wikipedia.org/wiki/Computer_Pioneer_Award>`_). Given that now
       it is *rarely* used to optimize black-box problems, it is **highly recommended** to first attempt other more
       advanced methods for large-scale black-box optimization (LSBBO).

       Since its original version needs `3**n - 1` samples for each iteration in the worst case, where `n` is the
       dimensionality of the problem. Such a worst-case complexity limits its applicability for LSBBO scenarios.
       Instead, here we use the **opportunistic** strategy for simplicity.
       See Algorithm 3 from `Torczon, 1997, SIAM-JO <https://epubs.siam.org/doi/abs/10.1137/S1052623493250780>`_.

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
                * 'max_runtime'              - maximal runtime (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`),
                * 'record_fitness'           - flag to record fitness list to output results (`bool`, default: `False`),
                * 'record_fitness_frequency' - function evaluations frequency of recording (`int`, default: `1000`),

                  * if `record_fitness` is set to `False`, it will be ignored,
                  * if `record_fitness` is set to `True` and it is set to 1, all fitness generated during optimization
                    will be saved into output results.

                * 'verbose'                  - flag to print verbose info during optimization (`bool`, default: `True`),
                * 'verbose_frequency'        - frequency of printing verbose info (`int`, default: `10`);
              and with three particular settings (`keys`):
                * 'x'     - initial (starting) point (`array_like`),
                * 'sigma' - initial (global) step-size (`float`),
                * 'gamma' - decreasing factor of step-size (`float`, default: `0.5`).

    Examples
    --------
    Use the Pattern Search optimizer `CS` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.ds.cs import CS
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3 * numpy.ones((2,)),
       ...            'sigma': 1.0,
       ...            'verbose_frequency': 500}
       >>> coordinate_search = CS(problem, options)  # initialize the optimizer class
       >>> results = coordinate_search.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"Coordinate Search: {results['n_function_evaluations']}, {results['best_so_far_y']}")
         * Generation 500: best_so_far_y 2.74643e+01, min(y) 1.98634e+03 & Evaluations 2017
         * Generation 1000: best_so_far_y 1.48952e+00, min(y) 3.66059e+04 & Evaluations 4033
       Coordinate Search: 5000, 0.1491367032979898

    Attributes
    ----------
    x     : `array_like`
            initial (starting) point.
    sigma : `float`
            initial (global) step-size.
    gamma : `float`
            decreasing factor of step-size.

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
        self.gamma = options.get('gamma', 0.5)  # decreasing factor of step-size
        assert self.gamma > 0.0, f'`self.gamma` == {self.gamma}, but should > 0.0.'

    def initialize(self, args=None, is_restart=False):
        x = self._initialize_x(is_restart)  # initial point
        y = self._evaluate_fitness(x, args)  # fitness
        return x, y

    def iterate(self, args=None, x=None, fitness=None):
        improved = False
        for i in range(self.ndim_problem):  # search along each coordinate
            for sgn in [-1, 1]:  # for two opponent directions
                if self._check_terminations():
                    return x
                xx = np.copy(x)
                xx[i] += sgn * self.sigma
                y = self._evaluate_fitness(xx, args)
                if self.record_fitness:
                    fitness.append(y)
                if y < self.best_so_far_y:
                    x = xx  # greedy / opportunistic
                    improved = True
                    break
        if not improved:  # decrease step-size if no improvement
            self.sigma *= self.gamma
        return x

    def restart_initialize(self, args=None, x=None, y=None, fitness=None):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self.n_restart += 1
            self.sigma = np.copy(self._sigma_bak)
            x, y = self.initialize(args, is_restart)
            fitness.append(y)
            self._fitness_list = [self.best_so_far_y]
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x, y = self.initialize(args)
        fitness.append(y)
        while True:
            x = self.iterate(args, x, fitness)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                x, y = self.restart_initialize(args, x, y, fitness)
        results = self._collect_results(fitness)
        return results
