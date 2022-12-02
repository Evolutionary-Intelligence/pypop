import numpy as np

from pypop7.optimizers.ga.ga import GA


class GL25(GA):
    """Global and Local genetic algorithm (GL25).

    .. note:: `25` means that 25 percentage of function evaluations (or runtime) are first used for *global* search
       while the remaining 75 percentage are then used for *local* search.

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
                * 'alpha'           - global step-size for crossover (`float`, default: `0.8`),
                * 'n_female_global' - number of female at global search stage (`int`, default: `200`),
                * 'n_male_global'   - number of male at global search stage (`int`, default: `400`),
                * 'n_female_local'  - number of female at local search stage (`int`, default: `5`),
                * 'n_male_local'    - number of male at local search stage (`int`, default: `100`),
                * 'p_global'        - percentage of global search stage (`float`, default: `0.25`),.

    Examples
    --------
    Use the Genetic Algorithm optimizer `GL25` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.ga.gl25 import GL25
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> gl25 = GL25(problem, options)  # initialize the optimizer class
       >>> results = gl25.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"GL25: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       GL25: 5000, 1.0505276479694516e-05

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/ytzffmbc>`_ for more details.

    Attributes
    ----------
    alpha           : `float`
                      global step-size for crossover.
    n_female_global : `int`
                      number of female at global search stage.
    n_female_local  : `int`
                      number of female at local search stage.
    n_individuals   : `int`
                      population size.
    n_male_global   : `int`
                      number of male at global search stage.
    n_male_local    : `int`
                      number of male at local search stage.
    p_global        : `float`
                      percentage of global search stage.

    References
    ----------
    García-Martínez, C., Lozano, M., Herrera, F., Molina, D. and Sánchez, A.M., 2008.
    Global and local real-coded genetic algorithms based on parent-centric crossover operators.
    European Journal of Operational Research, 185(3), pp.1088-1113.
    https://www.sciencedirect.com/science/article/abs/pii/S0377221706006308
    """
    def __init__(self, problem, options):
        GA.__init__(self, problem, options)
        self.alpha = options.get('alpha', 0.8)
        self.p_global = options.get('p_global', 0.25)  # percentage of global search stage
        self.n_female_global = options.get('n_female_global', 200)  # number of female at global search stage
        self.n_male_global = options.get('n_male_global', 400)  # number of male at global search stage
        self.n_female_local = options.get('n_female_local', 5)  # number of female at local search stage
        self.n_male_local = options.get('n_male_local', 100)  # number of male at local search stage
        self.n_individuals = int(np.maximum(self.n_male_global, self.n_male_local))
        self._assortative_mating = 5
        self._n_selected = np.zeros((self.n_individuals,))  # number of individuals selected as female
        # set maximum of function evaluations and runtime for global search stage
        self._max_fe_global = self.p_global*self.max_function_evaluations
        self._max_runtime_global = self.p_global*self.max_runtime

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # population
        y = np.empty((self.n_individuals,))  # fitness
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y

    def iterate(self, x=None, y=None, n_female=None, n_male=None, fitness=None, args=None):
        order = np.argsort(y)
        x_male, female = x[order[range(n_male)]], order[range(n_female)]
        x_female, _n_selected = x[female], self._n_selected[female]
        # use the uniform fertility selection (UFS) as female selection mechanism
        female = np.argmin(_n_selected)
        _n_selected[female] += 1
        self._n_selected[order[range(n_female)]] = _n_selected
        female = x_female[female]
        # use negative assortative mating (NAM) as male selection mechanism
        distances = np.empty((self._assortative_mating,))
        male = self.rng_optimization.choice(n_male, size=self._assortative_mating, replace=False)
        for i, m in enumerate(male):
            distances[i] = np.linalg.norm(female - x_male[m])
        male = x_male[male[np.argmax(distances)]]
        # use the parent-centric BLX crossover operator
        interval = np.abs(female - male)
        l, u = female - interval*self.alpha, female + interval*self.alpha
        xx = self.rng_optimization.uniform(np.clip(l, self.lower_boundary, self.upper_boundary),
                                           np.clip(u, self.lower_boundary, self.upper_boundary))
        yy = self._evaluate_fitness(xx, args)
        if self.saving_fitness:
            fitness.append(yy)
        # use the replace worst (RW) strategy
        if yy < y[order[-1]]:
            x[order[-1]], y[order[-1]], self._n_selected[order[-1]] = xx, yy, 0
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness, is_switch = GA.optimize(self, fitness_function), True
        x, y = self.initialize(args)
        if self.saving_fitness:
            fitness.extend(y)
        while True:
            if self.n_function_evaluations >= self._max_fe_global or self.runtime >= self._max_runtime_global:
                if is_switch:  # local search
                    init, is_switch = range(np.maximum(self.n_female_local, self.n_male_local)), False
                    x, y, self._n_selected = x[init], y[init], self._n_selected[init]
                x, y = self.iterate(x, y, self.n_female_local, self.n_male_local, fitness, args)
            else:  # global search
                x, y = self.iterate(x, y, self.n_female_global, self.n_male_global, fitness, args)
            if self._check_terminations():
                break
            self._print_verbose_info(y)
            self._n_generations += 1
        return self._collect_results(fitness)
