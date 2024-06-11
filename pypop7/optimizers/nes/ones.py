import numpy as np  # engine for numerical computing

from pypop7.optimizers.nes.nes import NES  # abstract class of Natural Evolution Strategies (NES) classes
from pypop7.optimizers.nes.sges import SGES  # Search Gradient-based Evolution Strategy (SGES) class


class ONES(SGES):
    """Original Natural Evolution Strategy (ONES).

    .. note:: Here we include `ONES` **mainly** for *benchmarking* and/or *theoretical* purpose. In practice,
       more advanced versions (e.g., `ENES`, `XNES`, `SNES`, and `R1NES`) should be first considered rather
       than the original version, which was first published in IEEE CEC-2008 by Schmidhuber's team. Simply
       speaking, the **parameterized** search distribution makes the mathematical derivation of the complex
       population update/evolution process possible and tractable, under mild assumptions. 

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

                  * If not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.
                * 'lr_mean'       - learning rate of distribution mean update (`float`, default: `1.0`),
                * 'lr_sigma'      - learning rate of global step-size adaptation (`float`, default: `1.0`).

    Examples
    --------
    Use the optimizer `ONES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.nes.ones import ONES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> ones = ONES(problem, options)  # initialize the optimizer class
       >>> results = ones.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"ONES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       ONES: 5000, 4.08973753355584e-05

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
    Beyer, H.G., 2023, July.
    `What you always wanted to know about evolution strategies, but never dared to ask.
    <https://doi.org/10.1145/3583133.3595041>`_
    In Proceedings of Companion Conference on Genetic and Evolutionary Computation (pp. 878-894). ACM.

    Beyer, H.G., 2014.
    `Convergence analysis of evolutionary algorithms that are based on the paradigm of information geometry.
    <https://doi.org/10.1162/EVCO_a_00132>`_
    Evolutionary Computation, 22(4), pp.679-709.

    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014.
    `Natural evolution strategies.
    <https://jmlr.org/papers/v15/wierstra14a.html>`_
    Journal of Machine Learning Research, 15(1), pp.949-980.

    Schaul, T., 2011.
    `Studies in continuous black-box optimization.
    <https://people.idsia.ch/~schaul/publications/thesis.pdf>`_
    Doctoral Dissertation, Technische Universität München.

    Please refer to the *official* Python source code from `PyBrain` (now not actively maintained):
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/nes.py
    """
    def __init__(self, problem, options):
        """Initialize all the hyper-parameters and also auxiliary class members.
        """
        SGES.__init__(self, problem, options)
        if options.get('lr_mean') is None:
            self.lr_mean = 1.0
        if options.get('lr_sigma') is None:
            self.lr_sigma = 1.0

    def _update_distribution(self, x=None, y=None, mean=None, cv=None):
        """Update the mean and covariance matrix of Gaussian search/sampling/mutation distribution.
        """
        # sort the offspring population for *maximization* (`-y`) rather than *minimization*
        order = np.argsort(-y)
        # ensure that the better an offspring, the larger its weight
        u = np.empty((self.n_individuals,))
        for i, o in enumerate(order):
            u[o] = self._u[i]
        # calculate the inverse of covariance matrix
        inv_cv = np.linalg.inv(cv)
        # calculate all derivatives w.r.t. both mean and covariance matrix (`+ 1` is a trick)
        phi = np.ones((self.n_individuals, self._n_distribution + 1))
        for k in range(self.n_individuals):  # for each offspring individual
            diff = x[k] - mean
            phi[k, :self.ndim_problem] = np.dot(inv_cv, diff)
            _grad_cv = 0.5 * (np.dot(np.dot(inv_cv, np.outer(diff, diff)), inv_cv) - inv_cv)
            phi[k, self.ndim_problem:-1] = self._triu2flat(np.dot(self._d_cv, _grad_cv + _grad_cv.T))
        grad = np.dot(np.linalg.pinv(phi), u)[:-1]  # `pinv` -> compute the (Moore-Penrose) pseudo-inverse
        # update the mean of Gaussian search/sampling/mutation distribution
        mean += self.lr_mean * grad[:self.ndim_problem]
        # update the covariance matrix of Gaussian search/sampling/mutation distribution
        self._d_cv += self.lr_sigma * self._flat2triu(grad[self.ndim_problem:])
        cv = np.dot(self._d_cv.T, self._d_cv)  # to recover covariance matrix
        self._n_generations += 1
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
            x, y, mean, cv = self._update_distribution(x, y, mean, cv)
            x, y, mean, cv = self.restart_reinitialize(x, y, mean, cv)
        return self._collect(fitness, y, mean)
