import numpy as np
import time

from pypop7.optimizers.core.optimizer import Optimizer
from pypop7.optimizers.sa.sa import SA
import scipy.optimize._dual_annealing as DUA


class DA(SA):
    """Dual Annealing(DA)

    .. note:: `"The algorithm is adapted from the dual annealing algorithm created by scipy.
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html

       This function derived from Generalized simulated annealing algorithm combines the generalization
       of CSA (Classical Simulated Annealing) and FSA (Fast Simulated Annealing) coupled to a strategy
       for applying a local search on accepted locations.

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
                * 'temperature'        - annealing temperature (`float`, 5230),
                * 'restart_temp_ratio' - annealing temperature (`float`, 2e-5),
                * 'max_iter'           - maximum of global search iteration (`int`, 1000),
                * 'visit'              - parameter for visit distribution (`float`, default: `2.62`),
                * 'accept'             - parameter for acceptance distribution. (`float`, default: `-5.0`),
                * 'if_local_search'    - whether use local search (`bool`, default: `True`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.sa.da import DA
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3*numpy.ones((2,))}
       >>> da = DA(problem, options)  # initialize the optimizer class
       >>> results = da.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"DA: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       DA: 5000, 9.478867754223762e-12

    Attributes
    ----------
    temperature        : `float`
                         Initial temperature.
    restart_temp_ratio : `float`
                         ratio that used for restart the loop.
    max_iter           : `int`
                         maximum of global search iteration.
    visit              : `float`
                         parameter for visit distribution.
    accept             : `float`
                         parameter for acceptance distribution.
    if_local_search    : `bool`
                         whether use local search.
    x                  : `array_like`
                         initial (starting) point.

    References
    ----------
    Xiang, Y., Sun, D. Y., Fan, W., & Gong, X. G., 1997.
    Generalized simulated annealing algorithm and its application to the Thomson model.
    Physics Letters A, 233(3), 216-220.
    https://www.sciencedirect.com/science/article/abs/pii/S037596019700474X

    Tsallis C., 1998.
    Possible generalization of Boltzmann-Gibbs statistics.
    Journal of Statistical Physics, 52, 479-487.
    https://link.springer.com/article/10.1007/BF01016429

    Xiang, Y., Gubian, S., Suomela, B., & Hoeng, J., 2013.
    Generalized simulated annealing for global optimization: the GenSA package.
    R J., 5(1), 13.
    https://svn.r-project.org/Rjournal/trunk/html/_site/archive/2013/RJ-2013-002/RJ-2013-002.pdf
    """
    def __init__(self, problem, options):
        SA.__init__(self, problem, options)
        self.temperature = options.get('temperature', 5230)  # annealing temperature
        self.restart_temp_ratio = options.get('restart_temperature_ratio', 2e-5)
        self.visit = options.get('visit', 2.62)
        self.accept = options.get('accept', -5.0)
        self.max_iter = options.get('max_iter', 1000)
        self.if_local_search = options.get('if_local_search', True)
        self.temperature_restart = self.temperature * self.restart_temp_ratio
        # Wrapper for the objective function
        self.func_wrapper = DUA.ObjectiveFunWrapper(self.fitness_function, self.max_function_evaluations)
        # Wrapper fot the minimizer
        bounds = list(zip(self.lower_boundary, self.upper_boundary))
        self.minimizer_wrapper = DUA.LocalSearchWrapper(bounds, self.func_wrapper)
        # Initialization of random Generator for reproducible runs if seed provided
        self.rand_state = DUA.check_random_state(self.seed_rng)
        # Initialization of the energy state
        self.energy_state = DUA.EnergyState(self.lower_boundary, self.upper_boundary, callback=None)
        self.t1 = np.exp((self.visit - 1) * np.log(2.0)) - 1.0

    def initialize(self, args=None):
        if self.x is None:  # starting point
            x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        else:
            x = np.copy(self.x)
        assert len(x) == self.ndim_problem
        self.energy_state.reset(self.func_wrapper, self.rand_state, x)
        # VisitingDistribution instance
        visit_dist = DUA.VisitingDistribution(self.lower_boundary, self.upper_boundary, self.visit, self.rand_state)
        # Strategy chain instance
        self.strategy_chain = DUA.StrategyChain(self.accept, visit_dist, self.func_wrapper,
                                                self.minimizer_wrapper, self.rand_state, self.energy_state)
        y = self._evaluate_fitness(x, args)
        return y

    def iterate(self, i):
        y = []
        s = float(i) + 2.0
        t2 = np.exp((self.visit - 1) * np.log(s)) - 1.0
        temp = self.temperature * self.t1 / t2
        if temp < self.temperature_restart:
            self.energy_state.reset(self.func_wrapper, self.rand_state)
            return True, None
        val = self.strategy_chain.run(i, temp)
        y.append(self.energy_state.current_energy)
        if self.energy_state.current_energy < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(self.energy_state.current_location),\
                                                     self.energy_state.current_energy
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += 1
        if val is not None:
            return True, y
        if self.if_local_search:
            val = self.strategy_chain.local_search()
            y.append(self.energy_state.current_energy)
            if self.energy_state.current_energy < self.best_so_far_y:
                self.best_so_far_x, self.best_so_far_y = np.copy(
                    self.energy_state.current_location), self.energy_state.current_energy
            self.time_function_evaluations += time.time() - self.start_function_evaluations
            self.n_function_evaluations += 1
            if val is not None:
                return True, y
        return False, y

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        y = self.initialize(args)
        self._print_verbose_info(fitness, y)
        while not self._check_terminations():
            for i in range(self.max_iter):
                if self._check_terminations():
                    break
                is_retart, y = self.iterate(i)
                if y is None:
                    break
                self._n_generations += 1
                self._print_verbose_info(fitness, y)
                if is_retart is True:
                    break
        return self._collect(fitness)
