import numpy as np

from pypop7.optimizers.ga.ga import GA


class HRCGA(GA):
    """HRCGA(Hybrid RCGA)

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
              and with the following particular settings (`keys`):
                * 'n_individuals' - population size (`int`, default: `100`),
                * 'alpha' - step size (`float`, default: `0.7`),
                * 'P_G' - part to run global RCGA (`float`, default: `0.25`),
                * 'N_GM' - best male individuals amount of global RCGA (`int`, default: `400`),
                * 'N_GF' - best female individuals amount of global RCGA (`int`, default: `200`),
                * 'N_LM' - best male individuals amount of local RCGA (`int`, default: `200`),
                * 'N_LF' - best female individuals amount of local RCGA (`int`, default: `200`),

    Examples
    --------
    Use the GA optimizer `HRCGA` to minimize the well-known test function
    `Rastrigin <http://en.wikipedia.org/wiki/Rastrigin_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rastrigin  # function to be minimized
       >>> from pypop7.optimizers.ga.hrcga import HRCGA
       >>> problem = {'fitness_function': rastrigin,  # define problem arguments
       ...            'ndim_problem': 25,
       ...            'lower_boundary': -5 * numpy.ones((100,)),
       ...            'upper_boundary': 5 * numpy.ones((100,))}
       >>> options = {'max_function_evaluations': 100000,  # set optimizer options
       ...            'n_individuals': 1000,
       ...            'seed_rng': 2022}
       >>> hrcga = HRCGA(problem, options)  # initialize the optimizer class
       >>> results = hrcga.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"HRCGA: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       HRCGA: 100000, 17.909247914222192

    Attributes
    ----------
    n_individuals : `int`
                    population size.
    alpha: `float`
                    step size.
    P_G: `float`
                    part to run global RCGA.
    N_GM : `int`
                    best male individuals amount of global RCGA.
    N_GF : `int`
                    best female individuals of amount global RCGA.
    N_LM : `int`
                    best male individuals of amount local RCGA.
    N_LF : `int`
                    best female individuals of amount local RCGA.

    References
    ----------
    C. G. Martinez, M. Lozano, F. Herrera, D. Molina, A. M. Sanchez
    Global and local real-coded genetic algorithms based on parent-centric crossover operators
    European Journal of Operational Research 185 (2008) 1088–1113
    """
    def __init__(self, problem, options):
        GA.__init__(self, problem, options)
        self.alpha = options.get('alpha', 0.7)
        self.P_G = options.get('P_G', 0.25)
        self.N_GM = options.get('N_GM', 400)
        self.N_GF = options.get('N_GF', 200)
        self.N_LM = options.get('N_LM', 100)
        self.N_LF = options.get('N_LF', 5)
        self.n_ass = 5
        self.global_function_evaluations = self.P_G * self.max_function_evaluations
        self.global_indices = np.arange(0, self.N_GM)
        self.local_indices = np.arange(0, self.N_LM)

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # population
        y = np.empty((self.n_individuals,))  # fitness
        sel_time = np.zeros((self.n_individuals,))
        x_lm = np.empty((self.N_LM, self.ndim_problem))
        x_lf = np.empty((self.N_LF, self.ndim_problem))
        x_gm = np.empty((self.N_GM, self.ndim_problem))
        x_gf = np.empty((self.N_GF, self.ndim_problem))
        g_time = np.empty((self.N_GF,))
        l_time = np.empty((self.N_LF,))
        for i in range(self.n_individuals):
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y, x_lm, x_lf, x_gm, x_gf, sel_time, l_time, g_time

    def ufs(self, x=None, times=None):
        min_time = np.argmin(times)
        return x[min_time], min_time

    def nam(self, x=None, pf=None):
        if self.n_function_evaluations < self.global_function_evaluations:
            places = np.random.choice(self.global_indices, self.n_ass)
        else:
            places = np.random.choice(self.local_indices, self.n_ass)
        similarities = np.empty((self.n_ass,))
        for i in range(self.n_ass):
            similarities[i] = np.linalg.norm(pf - x[places[i]])
        minplace = np.argmax(similarities)
        return x[places[minplace]]

    def iterate(self, x=None, y=None, x_m=None, x_f=None, sel_time=None, times=None, fitness=None, args=None):
        order = np.argsort(y)
        if self.n_function_evaluations < self.global_function_evaluations:
            for i in range(self.N_GM):
                x_m[i] = x[order[i]].copy()
            for i in range(self.N_GF):
                x_f[i] = x[order[i]].copy()
                times[i] = sel_time[order[i]].copy()
        else:
            for i in range(self.N_LM):
                x_m[i] = x[order[i]].copy()
            for i in range(self.N_LF):
                x_f[i] = x[order[i]].copy()
                times[i] = sel_time[order[i]].copy()
        # select parents
        pf, min_time = self.ufs(x_f, times)
        sel_time[order[min_time]] += 1
        pm = self.nam(x_m, pf)

        # create offspring
        I = np.abs(pf - pm)
        l = pf - I * self.alpha
        l = np.clip(l, self.lower_boundary, self.upper_boundary)
        u = pf + I * self.alpha
        u = np.clip(u, self.lower_boundary, self.upper_boundary)
        off_spring = self.rng_initialization.uniform(l, u, size=(self.ndim_problem,))
        off_y = self._evaluate_fitness(off_spring, args)
        if off_y < y[order[-1]]:
            x[order[-1]], y[order[-1]], sel_time[order[-1]] = off_spring, off_y, 0
        if self.saving_fitness:
            fitness.append(off_y)
        return x, y, sel_time, fitness

    def optimize(self, fitness_function=None, args=None):
        fitness = GA.optimize(self, fitness_function)
        x, y, x_lm, x_lf, x_gm, x_gf, sel_time, l_time, g_time = self.initialize(args=None)
        fitness.extend(y)
        while True:
            if self.n_function_evaluations < self.global_function_evaluations:
                x, y, time, fitness = self.iterate(x, y, x_gm, x_gf, sel_time, g_time, fitness, args)
            else:
                x, y, time, fitness = self.iterate(x, y, x_lm, x_lf, sel_time, l_time, fitness, args)
            if self._check_terminations():
                break
            self._print_verbose_info(y)
            self._n_generations += 1
        return self._collect_results(fitness)
