import numpy as np

from pypop7.optimizers.cem.cem import CEM


class DSCEM(CEM):
    """Dynamic Smoothing Cross-Entropy Method (DSCEM).

    .. note:: `DSCEM` uses the *dynamic* smoothing strategy to update the mean and std of Gaussian search
       (mutation/sampling) distribution in an online fashion.

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
                * 'sigma'         - initial global step-size, aka mutation strength (`float`),
                * 'mean'          - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'n_individuals' - offspring population size (`int`, default: `1000`),
                * 'n_parents'     - parent population size (`int`, default: `200`),
                * 'alpha'         - smoothing factor of mean of Gaussian search distribution (`float`, default: `0.8`),
                * 'beta'          - smoothing factor of individual step-sizes (`float`, default: `0.7`),
                * 'q'             - decay factor of smoothing individual step-sizes (`float`, default: `5.0`).

    Examples
    --------
    Use the optimizer `DSCEM` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.cem.dscem import DSCEM
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 100,
       ...            'lower_boundary': -5*numpy.ones((100,)),
       ...            'upper_boundary': 5*numpy.ones((100,))}
       >>> options = {'max_function_evaluations': 1000000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'sigma': 0.3}  # the global step-size may need to be tuned for better performance
       >>> dscem = DSCEM(problem, options)  # initialize the optimizer class
       >>> results = dscem.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"DSCEM: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       DSCEM: 1000000, 158.66725776324424

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/43f6kjee>`_ for more details.

    Attributes
    ----------
    alpha         : `float`
                    smoothing factor of mean of Gaussian search distribution.
    beta          : `float`
                    smoothing factor of individual step-sizes.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.
    q             : `float`
                    decay factor of smoothing individual step-sizes.
    sigma         : `float`
                    initial global step-size, aka mutation strength.

    References
    ----------
    Kroese, D.P., Porotsky, S. and Rubinstein, R.Y., 2006.
    The cross-entropy method for continuous multi-extremal optimization.
    Methodology and Computing in Applied Probability, 8(3), pp.383-407.
    https://link.springer.com/article/10.1007/s11009-006-9753-0
    (See [Appendix B Main CE Program] for the official Matlab code.)

    De Boer, P.T., Kroese, D.P., Mannor, S. and Rubinstein, R.Y., 2005.
    A tutorial on the cross-entropy method.
    Annals of Operations Research, 134(1), pp.19-67.
    https://link.springer.com/article/10.1007/s10479-005-5724-z
    """
    def __init__(self, problem, options):
        CEM.__init__(self, problem, options)
        self.alpha = options.get('alpha', 0.8)  # smoothing factor of mean of Gaussian search distribution
        self.beta = options.get('beta', 0.7)  # smoothing factor of individual step-sizes
        self.q = options.get('q', 5.0)  # decay factor of smoothing individual step-sizes

    def initialize(self, is_restart=False):
        mean = self._initialize_mean(is_restart)
        x = np.empty((self.n_individuals, self.ndim_problem))  # samples (population)
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return mean, x, y

    def iterate(self, mean=None, x=None, y=None, args=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            x[i] = mean + self._sigmas*self.rng_optimization.standard_normal((self.ndim_problem,))
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y

    def _update_parameters(self, mean=None, x=None, y=None):
        xx = x[np.argsort(y)[:self.n_parents]]
        mean = self.alpha*np.mean(xx, axis=0) + (1.0-self.alpha)*mean
        b_mod = self.beta - self.beta*np.power(1.0 - 1.0/self._n_generations, self.q)
        self._sigmas = b_mod*np.std(xx, axis=0) + (1.0 - b_mod)*self._sigmas
        return mean

    def optimize(self, fitness_function=None, args=None):
        fitness = CEM.optimize(self, fitness_function)
        mean, x, y = self.initialize()
        while True:
            x, y = self.iterate(mean, x, y, args)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._print_verbose_info(y)
            self._n_generations += 1
            mean = self._update_parameters(mean, x, y)
        return self._collect_results(fitness, mean)
