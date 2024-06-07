import numpy as np  # engine for numerical computing

from pypop7.optimizers.nes.nes import NES  # abstract class of Natural Evolution Strategies (NES) classes


class SGES(NES):
    """Search Gradient-based Evolution Strategy (SGES).

    .. note:: Here we include `SGES` (also called **vanilla version** of `NES`) **only** for *theoretical*
       and *educational* purposes, since in practice advanced versions (e.g., `ENES`, `XNES`, `SNES`, and
       `R1NES`) are more preferred than `SGES` in most cases.

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
                * 'lr_mean'       - learning rate of distribution mean update (`float`, default: `0.01`),
                * 'lr_sigma'      - learning rate of global step-size adaptation (`float`, default: `0.01`).

    Examples
    --------
    Use the optimizer `SGES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.nes.sges import SGES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,))}
       >>> sges = SGES(problem, options)  # initialize the optimizer class
       >>> results = sges.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"SGES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       SGES: 5000, 0.01906602832229609

    Attributes
    ----------
    lr_mean       : `float`
                    learning rate of distribution mean update (should `> 0.0`).
    lr_sigma      : `float`
                    learning rate of global step-size adaptation (should `> 0.0`).
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search/sampling/mutation distribution.
                    If not given, it will draw a random sample from the uniform distribution whose search
                    range is bounded by `problem['lower_boundary']` and `problem['upper_boundary']`, by
                    default.
    n_individuals : `int`
                    number of offspring/descendants, aka offspring population size (should `> 0`).
    n_parents     : `int`
                    number of parents/ancestors, aka parental population size (should `> 0`).

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

    Please refer to the *official* Python source code from `PyBrain` (now not actively maintained):
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/ves.py
    """
    def __init__(self, problem, options):
        """Initialize all the hyper-parameters and also auxiliary class members.
        """
        options['n_individuals'] = options.get('n_individuals', 100)
        options['sigma'] = np.Inf  # but not used for `SGES` here
        NES.__init__(self, problem, options)
        if self.lr_mean is None:
            self.lr_mean = 0.01
        assert self.lr_mean > 0.0, f'`self.lr_mean` = {self.lr_mean}, but should > 0.0.'
        if self.lr_sigma is None:
            self.lr_sigma = 0.01
        assert self.lr_sigma > 0.0, f'`self.lr_sigma` = {self.lr_sigma}, but should > 0.0.'
        # set parameter number of Gaussian search/sampling/mutation distribution
        self._n_distribution = int(self.ndim_problem + self.ndim_problem * (self.ndim_problem + 1) / 2)
        self._d_cv = None  # all derivatives w.r.t. covariance matrix

    def initialize(self, is_restart=False):
        """Initialize the offspring population, their fitness, mean and covariance matrix of Gaussian
           search/sampling/mutation distribution.
        """
        NES.initialize(self)
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        y = np.empty((self.n_individuals,))  # fitness (no evaluation when initialization)
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search/sampling/mutation distribution
        cv = np.eye(self.ndim_problem)  # covariance matrix of Gaussian search/sampling/mutation distribution
        self._d_cv = np.eye(self.ndim_problem)  # all derivatives w.r.t. covariance matrix
        return x, y, mean, cv

    def iterate(self, x=None, y=None, mean=None, args=None):
        """Iterate the generation and fitness evaluation process of the offspring population.
        """
        for k in range(self.n_individuals):  # for each offspring individual
            if self._check_terminations():
                return x, y
            # generate each offspring individual according to Gaussian search/sampling/mutation distribution
            x[k] = mean + np.dot(self._d_cv.T, self.rng_optimization.standard_normal((self.ndim_problem,)))
            # evaluate the fitness of each offspring individual according to objective function
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _triu2flat(self, cv):
        """Convert the upper-triangular matrix to an entirely flat vector.
        """
        v = np.zeros((self._n_distribution - self.ndim_problem,))
        s, e = 0, self.ndim_problem  # starting and ending index
        for r in range(self.ndim_problem):
            v[s:e] = cv[r, r:]
            s, e = e, e + (self.ndim_problem - (r + 1))
        return v

    def _flat2triu(self, g):
        """Convert the entirely flat vector to an upper-triangular matrix.
        """
        cv = np.zeros((self.ndim_problem, self.ndim_problem))
        s, e = 0, self.ndim_problem  # starting and ending index
        for r in range(self.ndim_problem):
            cv[r, r:] = g[s:e]
            s, e = e, e + (self.ndim_problem - (r + 1))
        return cv

    def _update_distribution(self, x=None, y=None, mean=None, cv=None):
        """Update the mean and covariance matrix of Gaussian search/sampling/mutation distribution.
        """
        # sort the offspring population for *maximization* rather than *minimization* and
        order = np.argsort(-y)
        # ensure that the better an offspring, the larger its weight
        u = np.empty((self.n_individuals,))
        for i, o in enumerate(order):
            u[o] = self._u[i]
        # calculate the inverse of covariance matrix
        inv_cv = np.linalg.inv(cv)
        # calculate all derivatives w.r.t. both mean and covariance matrix
        phi = np.zeros((self.n_individuals, self._n_distribution))
        # calculate all derivatives w.r.t. mean for all offspring
        phi[:, :self.ndim_problem] = np.dot(inv_cv, (x - mean).T).T
        # calculate all derivatives w.r.t. covariance matrix for all offspring
        grad_cv = np.empty((self.n_individuals, self._n_distribution - self.ndim_problem))
        for k in range(self.n_individuals):
            diff = x[k] - mean
            _grad_cv = 0.5 * (np.dot(np.dot(inv_cv, np.outer(diff, diff)), inv_cv) - inv_cv)
            grad_cv[k] = self._triu2flat(np.dot(self._d_cv, (_grad_cv + _grad_cv.T)))
        phi[:, self.ndim_problem:] = grad_cv
        # use *fitness baseline* to reduce estimation variance rather than directly using
        # grad = np.sum(phi * (np.outer(u, np.ones((self._n_distribution,)))), 0)
        phi_square = phi * phi  # dynamic base
        grad = np.sum(phi * (np.outer(u, np.ones((self._n_distribution,))) - np.dot(
            u, phi_square) / np.dot(np.ones((self.n_individuals,)), phi_square)), 0)
        # update the mean of Gaussian search/sampling/mutation distribution
        mean += self.lr_mean * grad[:self.ndim_problem]
        # update the covariance matrix of Gaussian search/sampling/mutation distribution
        self._d_cv += self.lr_sigma * self._flat2triu(grad[self.ndim_problem:])
        cv = np.dot(self._d_cv.T, self._d_cv)  # to recover covariance matrix
        self._n_generations += 1
        return mean, cv

    def restart_reinitialize(self, x=None, y=None, mean=None, cv=None):
        """Restart and re-initialize the optimization/evolution process, if needed.
        """
        if self.is_restart and NES.restart_reinitialize(self, y):
            x, y, mean, cv = self.initialize(True)
        return x, y, mean, cv

    def optimize(self, fitness_function=None, args=None):
        """Run the optimization/evolution process for all generations (iterations).
        """
        fitness = NES.optimize(self, fitness_function)  # to store all fitness generated during optimization
        x, y, mean, cv = self.initialize()
        while True:
            x, y = self.iterate(x, y, mean, args)
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            mean, cv = self._update_distribution(x, y, mean, cv)
            x, y, mean, cv = self.restart_reinitialize(x, y, mean, cv)
        return self._collect(fitness, y, mean)
