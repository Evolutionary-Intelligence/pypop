import numpy as np  # engine for numerical computing

from pypop7.optimizers.nes.nes import NES


class SNES(NES):
    """Separable Natural Evolution Strategies (SNES).

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
                * 'n_individuals' - number of offspring/descendants, aka offspring population size (`int`),
                * 'n_parents'     - number of parents/ancestors, aka parental population size (`int`),
                * 'mean'          - initial (starting) point (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'sigma'         - initial global step-size, aka mutation strength (`float`).

    Examples
    --------
    Use the optimizer `SNES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.nes.snes import SNES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> snes = SNES(problem, options)  # initialize the optimizer class
       >>> results = snes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"SNES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       SNES: 5000, 0.49730042657448875

    Attributes
    ----------
    lr_cv         : `float`
                    learning rate of covariance matrix adaptation.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search/sampling/mutation distribution.
    n_individuals : `int`
                    number of offspring/descendants, aka offspring population size.
    n_parents     : `int`
                    number of parents/ancestors, aka parental population size.
    sigma         : `float`
                    global step-size, aka mutation strength (i.e., overall std of Gaussian search distribution).

    References
    ----------
    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014.
    `Natural evolution strategies.
    <https://jmlr.org/papers/v15/wierstra14a.html>`_
    Journal of Machine Learning Research, 15(1), pp.949-980.
    
    Schaul, T., 2011.
    `Studies in continuous black-box optimization.
    <https://people.idsia.ch/~schaul/publications/thesis.pdf>`_ 
    Doctoral Dissertation, Technische Universität München.

    Schaul, T., Glasmachers, T. and Schmidhuber, J., 2011, July.
    `High dimensions and heavy tails for natural evolution strategies.
    <https://dl.acm.org/doi/abs/10.1145/2001576.2001692>`_
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 845-852). ACM.

    See the official Python source code from PyBrain:
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/snes.py
    """
    def __init__(self, problem, options):
        NES.__init__(self, problem, options)
        self.lr_cv = 0.6*(3.0 + np.log(self.ndim_problem))/3.0/np.sqrt(self.ndim_problem)

    def initialize(self, is_restart=False):
        s = np.empty((self.n_individuals, self.ndim_problem))  # noise of offspring population
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        d = self.sigma*np.ones((self.ndim_problem,))  # individual step-sizes
        self._w = np.maximum(0.0, np.log(self.n_individuals/2.0 + 1.0) - np.log(
            self.n_individuals - np.arange(self.n_individuals)))
        return s, y, mean, d

    def iterate(self, s=None, y=None, mean=None, d=None, args=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return s, y
            s[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            y[k] = self._evaluate_fitness(mean + d*s[k], args)
        return s, y

    def _update_distribution(self, s=None, y=None, mean=None, d=None):
        order = np.argsort(-y)
        u = np.empty((self.n_individuals,))
        for i, o in enumerate(order):
            u[o] = self._w[i]
        u = u/np.sum(u) - 1.0/self.n_individuals
        mean += d*np.dot(u, s)
        d *= np.exp(0.5*self.lr_cv*np.dot(u, [np.square(k) - 1.0 for k in s]))
        self._n_generations += 1
        return mean, d

    def restart_reinitialize(self, s=None, y=None, mean=None, d=None):
        if self.is_restart and NES.restart_reinitialize(self, y):
            s, y, mean, d = self.initialize(True)
        return s, y, mean, d

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = NES.optimize(self, fitness_function)
        s, y, mean, d = self.initialize()
        while True:
            s, y = self.iterate(s, y, mean, d, args)
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            mean, d = self._update_distribution(s, y, mean, d)
            s, y, mean, d = self.restart_reinitialize(s, y, mean, d)
        return self._collect(fitness, y, mean)
