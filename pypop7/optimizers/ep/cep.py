import numpy as np

from pypop7.optimizers.ep.ep import EP


class CEP(EP):
    """Classical Evolutionary Programming with self-adaptive mutation (CEP).

    .. note:: To obtain satisfactory performance for large-scale black-box optimization, the number of
       offspring may need to be carefully tuned.

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
              and with five particular settings (`keys`):
                * 'n_individuals'  - number of offspring, offspring population size (`int`),
                * 'sigma'          - initial global step-size (σ), mutation strength (`float`),
                * 'q'              - number of opponents for pairwise comparisons (`int`, default: `10`),
                * 'tau'            - learning rate of individual step-sizes (`float`, default:
                  `1.0 / np.sqrt(2.0*np.sqrt(self.ndim_problem))`),
                * 'tau_apostrophe' - learning rate of individual step-sizes (`float`, default:
                  `1.0 / np.sqrt(2.0*self.ndim_problem)`.

    Examples
    --------
    Use the EP optimizer `CEP` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.ep.cep import CEP
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'sigma': 0.1}
       >>> cep = CEP(problem, options)  # initialize the optimizer class
       >>> results = cep.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CEP: {results['n_function_evaluations']}, {results['best_so_far_y']}")
         * Generation 10: best_so_far_y 2.92722e-01, min(y) 2.92722e-01 & Evaluations 1100
         * Generation 20: best_so_far_y 5.07126e-02, min(y) 5.07126e-02 & Evaluations 2100
         * Generation 30: best_so_far_y 1.65202e-03, min(y) 1.65202e-03 & Evaluations 3100
         * Generation 40: best_so_far_y 1.65202e-03, min(y) 1.65202e-03 & Evaluations 4100
       CEP: 5000, 0.001652015737093122

    For its correctness checking, refer to `this code-based repeatability report
    <https://tinyurl.com/b9vpmfdv>`_ for more details.

    Attributes
    ----------
    n_individuals  : `int`
                     number of offspring, population size.
    sigma          : `float`
                     initial global step-size, mutation strength.
    q              : `int`
                     number of opponents for pairwise comparisons。
    tau            : `float`
                     learning rate of individual step-sizes.
    tau_apostrophe : `float`
                     learning rate of individual step-sizes.

    References
    ----------
    Yao, X., Liu, Y. and Lin, G., 1999.
    Evolutionary programming made faster.
    IEEE Transactions on Evolutionary Computation, 3(2), pp.82-102.
    https://ieeexplore.ieee.org/abstract/document/771163

    Bäck, T. and Schwefel, H.P., 1993.
    An overview of evolutionary algorithms for parameter optimization.
    Evolutionary Computation, 1(1), pp.1-23.
    https://direct.mit.edu/evco/article-abstract/1/1/1/1092/An-Overview-of-Evolutionary-Algorithms-for
    """
    def __init__(self, problem, options):
        EP.__init__(self, problem, options)
        self.sigma = options.get('sigma')  # initial global step-size
        self.q = options.get('q', 10)  # number of opponents for pairwise comparisons
        # two learning rate factors of individual step-sizes
        self.tau = options.get('tau', 1.0 / np.sqrt(2.0*np.sqrt(self.ndim_problem)))
        self.tau_apostrophe = options.get('tau_apostrophe', 1.0 / np.sqrt(2.0*self.ndim_problem))

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))
        sigmas = self.sigma*np.ones((self.n_individuals, self.ndim_problem))  # eta (η)
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        offspring_x = np.empty((self.n_individuals, self.ndim_problem))
        offspring_sigmas = np.empty((self.n_individuals, self.ndim_problem))  # eta (η)
        offspring_y = np.empty((self.n_individuals,))
        return x, sigmas, y, offspring_x, offspring_sigmas, offspring_y

    def iterate(self, x=None, sigmas=None, y=None,
                offspring_x=None, offspring_sigmas=None, offspring_y=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, sigmas, y, offspring_x, offspring_sigmas, offspring_y
            for j in range(self.ndim_problem):
                offspring_sigmas[i][j] = sigmas[i][j]*np.exp(
                    self.tau_apostrophe*self.rng_optimization.standard_normal() +
                    self.tau*self.rng_optimization.standard_normal())
                offspring_x[i][j] = x[i][j] + offspring_sigmas[i][j]*self.rng_optimization.standard_normal()
            offspring_y[i] = self._evaluate_fitness(offspring_x[i])
        new_x = np.vstack((offspring_x, x))
        new_sigmas = np.vstack((offspring_sigmas, sigmas))
        new_y = np.hstack((offspring_y, y))
        n_win = np.zeros((2*self.n_individuals,))  # number of win
        for i in range(2*self.n_individuals):
            for j in self.rng_optimization.integers(2*self.n_individuals, size=self.q):
                if new_y[i] <= new_y[j]:
                    n_win[i] += 1
        order = np.argsort(-n_win)
        for i in range(self.n_individuals):
            x[i] = new_x[order[i]]
            sigmas[i] = new_sigmas[order[i]]
            y[i] = new_y[order[i]]
        return x, sigmas, y, offspring_x, offspring_sigmas, offspring_y

    def optimize(self, fitness_function=None, args=None):
        fitness = EP.optimize(self, fitness_function)
        x, sigmas, y, offspring_x, offspring_sigmas, offspring_y = self.initialize(args)
        fitness.extend(y)
        while True:
            x, sigmas, y, offspring_x, offspring_sigmas, offspring_y = self.iterate(
                x, sigmas, y, offspring_x, offspring_sigmas, offspring_y)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        return self._collect_results(fitness)
