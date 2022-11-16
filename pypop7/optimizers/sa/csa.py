import numpy as np

from pypop7.optimizers.rs.rs import RS


class CSA(RS):
    """Corana et al.' Simulated Annealing (CSA).

    .. note:: "The algorithm is essentially an iterative random search procedure with adaptive moves along
       the coordinate directions."---`[Corana et al., 1986, ACM-TOMS] <https://dl.acm.org/doi/abs/10.1145/29380.29864>`_

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
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`),
              and with the following particular settings (`keys`):
                * 'sigma'                    - initial global step-size (`float`),
                * 'temperature'              - annealing temperature (`float`),
                * 'n_sv'                     - frequency of step variation (`int`, default: `20`),
                * 'c'                        - factor of step variation (`float`, default: `2.0`),
                * 'n_tr'                     - frequency of temperature reduction (`int`, default:
                                               `np.maximum(100, 5*self.ndim_problem)`),
                * 'f_tr'                     - factor of temperature reduction (`int`, default: `0.85`).

    Examples
    --------
    Use the Simulated Annealing optimizer `CSA` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.sa.csa import CSA
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3 * numpy.ones((2,)),
       ...            'sigma': 1.0,
       ...            'temperature': 100}
       >>> csa = CSA(problem, options)  # initialize the optimizer class
       >>> results = csa.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CSA: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CSA: 5000, 0.0023146719686626344

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/bdd62drw>`_ for more details.

    References
    ----------
    Corana, A., Marchesi, M., Martini, C. and Ridella, S., 1987.
    Minimizing multimodal functions of continuous variables with the "simulated annealing" algorithm.
    ACM Transactions on Mathematical Software, 13(3), pp.262-280.
    https://dl.acm.org/doi/abs/10.1145/29380.29864
    https://dl.acm.org/doi/10.1145/66888.356281

    Kirkpatrick, S., Gelatt, C.D. and Vecchi, M.P., 1983.
    Optimization by simulated annealing.
    Science, 220(4598), pp.671-680.
    https://science.sciencemag.org/content/220/4598/671
    """
    def __init__(self, problem, options):
        RS.__init__(self, problem, options)
        self.sigma = options.get('sigma')
        self.v = self.sigma*np.ones((self.ndim_problem,))  # step vector
        self.temperature = options.get('temperature')  # temperature
        self.n_sv = options.get('n_sv', 20)  # frequency of step variation (N_S)
        c = options.get('c', 2.0)  # factor of step variation
        self.f_sv = c*np.ones(self.ndim_problem,)
        # set frequency of temperature reduction (N_T)
        self.n_tr = options.get('n_tr', np.maximum(100, 5*self.ndim_problem))
        self.f_tr = options.get('r_T', 0.85)  # factor of temperature reduction (r_T)
        self._sv = np.zeros((self.ndim_problem,))  # for step variation
        self.parent_x = None
        self.parent_y = None

    def initialize(self, args=None):
        if self.x is None:  # starting point
            x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        else:
            x = np.copy(self.x)
        y = self._evaluate_fitness(x, args)
        self.parent_x, self.parent_y = np.copy(x), np.copy(y)
        return y

    def iterate(self, args=None):  # to perform a cycle of random moves
        fitness = []
        for h in np.arange(self.ndim_problem):
            if self._check_terminations():
                break
            x = np.copy(self.parent_x)
            search_range = (np.maximum(self.parent_x[h] - self.v[h], self.lower_boundary[h]),
                            np.minimum(self.parent_x[h] + self.v[h], self.upper_boundary[h]))
            x[h] = self.rng_optimization.uniform(search_range[0], search_range[1])
            y = self._evaluate_fitness(x, args)
            if self.saving_fitness:
                fitness.append(y)
            self._n_generations += 1
            self._print_verbose_info(y)
            diff = self.parent_y - y
            if (diff >= 0) or (self.rng_optimization.random() < np.exp(diff/self.temperature)):
                self.parent_x, self.parent_y = np.copy(x), np.copy(y)
                self._sv[h] += 1
        return fitness

    def _adjust_step_vector(self):
        for u in range(self.ndim_problem):
            if self._sv[u] > 0.6*self.n_sv:
                self.v[u] *= 1.0 + self.f_sv[u]*(self._sv[u]/self.n_sv - 0.6)/0.4
            elif self._sv[u] < 0.4*self.n_sv:
                self.v[u] /= 1.0 + self.f_sv[u]*(0.4 - self._sv[u]/self.n_sv)/0.4
        self._sv = np.zeros((self.ndim_problem,))

    def optimize(self, fitness_function=None, args=None):
        super(RS, self).optimize(fitness_function)
        fitness = [self.initialize(args)]  # to store all fitness generated during search
        self._print_verbose_info(fitness[0])
        while not self._check_terminations():
            for m in range(self.n_tr):
                if self._check_terminations():
                    break
                for j in range(self.n_sv):
                    if self._check_terminations():
                        break
                    y = self.iterate(args)
                    if self.saving_fitness:
                        fitness.extend(y)
                self._adjust_step_vector()
            self.temperature *= self.f_tr  # temperature reducing
            self.parent_x, self.parent_y = np.copy(self.best_so_far_x), np.copy(self.best_so_far_y)
        results = RS._collect_results(self, fitness)
        results['v'] = np.copy(self.v)
        return results
