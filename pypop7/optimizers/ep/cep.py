import numpy as np

from pypop7.optimizers.ep.ep import EP


class CEP(EP):
    """Classical Evolutionary Programming with self-adaptive mutation (CEP).

    .. note:: To obtain satisfactory performance for large-scale black-box optimization, the number of
       offspring (`n_individuals`) and also initial global step-size (`sigma`) may need to be **carefully**
       tuned (e.g. via manual trial-and-error or automatical hyper-parameter optimization).

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
                * 'sigma'          - initial global step-size, aka mutation strength (`float`),
                * 'n_individuals'  - number of offspring, aka offspring population size (`int`, default: `100`),
                * 'q'              - number of opponents for pairwise comparisons (`int`, default: `10`),
                * 'tau'            - learning rate of individual step-sizes self-adaptation (`float`, default:
                  `1.0/np.sqrt(2.0*np.sqrt(self.ndim_problem))`),
                * 'tau_apostrophe' - learning rate of individual step-sizes self-adaptation (`float`, default:
                  `1.0/np.sqrt(2.0*self.ndim_problem)`.

    Examples
    --------
    Use the Evolutionary Programming optimizer `CEP` to minimize the well-known test function
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
       CEP: 5000, 0.3544823323771589

    For its correctness checking, refer to `this code-based repeatability report
    <https://tinyurl.com/b9vpmfdv>`_ for more details.

    Attributes
    ----------
    n_individuals  : `int`
                     number of offspring, aka offspring population size.
    q              : `int`
                     number of opponents for pairwise comparisons.
    sigma          : `float`
                     initial global step-size, aka mutation strength.
    tau            : `float`
                     learning rate of individual step-sizes self-adaptation.
    tau_apostrophe : `float`
                     learning rate of individual step-sizes self-adaptation.

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
        self.q = options.get('q', 10)  # number of opponents for pairwise comparisons
        # set two learning-rates of individual step-sizes adaptation
        self.tau = options.get('tau', 1.0/np.sqrt(2.0*np.sqrt(self.ndim_problem)))
        self.tau_apostrophe = options.get('tau_apostrophe', 1.0/np.sqrt(2.0*self.ndim_problem))

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))
        sigmas = self.sigma*np.ones((self.n_individuals, self.ndim_problem))  # eta
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        xx = np.empty((self.n_individuals, self.ndim_problem))
        ss = np.empty((self.n_individuals, self.ndim_problem))  # eta
        yy = np.empty((self.n_individuals,))
        return x, sigmas, y, xx, ss, yy

    def iterate(self, x=None, sigmas=None, y=None, xx=None, ss=None, yy=None, args=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, sigmas, y, xx, ss, yy
            # base = self.rng_optimization.standard_normal()
            # ss[i] = sigmas[i]*np.exp(self.tau_apostrophe*base + self.tau*self.rng_optimization.standard_normal(
            #     size=(self.ndim_problem,)))
            ss[i] = sigmas[i]*np.exp(self.tau_apostrophe*self.rng_optimization.standard_normal(
                size=(self.ndim_problem,)) + self.tau*self.rng_optimization.standard_normal(
                size=(self.ndim_problem,)))
            xx[i] = x[i] + ss[i]*self.rng_optimization.standard_normal(size=(self.ndim_problem,))
            yy[i] = self._evaluate_fitness(xx[i], args)
        new_x = np.vstack((xx, x))
        new_sigmas = np.vstack((ss, sigmas))
        new_y = np.hstack((yy, y))
        n_win = np.zeros((2*self.n_individuals,))  # number of win
        for i in range(2*self.n_individuals):
            for j in self.rng_optimization.choice(np.setdiff1d(range(2*self.n_individuals), i),
                                                  size=self.q, replace=False):
                if new_y[i] < new_y[j]:
                    n_win[i] += 1
        order = np.argsort(-n_win)[:self.n_individuals]
        x[:self.n_individuals] = new_x[order]
        sigmas[:self.n_individuals] = new_sigmas[order]
        y[:self.n_individuals] = new_y[order]
        self._n_generations += 1
        return x, sigmas, y, xx, ss, yy

    def optimize(self, fitness_function=None, args=None):
        fitness, is_initialization = EP.optimize(self, fitness_function), True
        x, sigmas, y, xx, ss, yy = None, None, None, None, None, None
        while True:
            if is_initialization:
                x, sigmas, y, xx, ss, yy = self.initialize(args)
                is_initialization = False
            else:
                x, sigmas, y, xx, ss, yy = self.iterate(x, sigmas, y, xx, ss, yy, args)
            if self.saving_fitness:
                fitness.extend(y)
            self._print_verbose_info(y)
            if self._check_terminations():
                break
        return self._collect_results(fitness)
