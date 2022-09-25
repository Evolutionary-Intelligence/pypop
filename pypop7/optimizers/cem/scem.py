import numpy as np

from pypop7.optimizers.cem.cem import CEM


class SCEM(CEM):
    """Standard Cross-Entropy Method (SCEM).

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
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'mean'          - initial mean (`array_like`),
                * 'sigma'         - initial global step-size (`float`),
                * 'n_individuals' - offspring population size (`int`, default: `1000`),
                * 'n_parents'     - parent population size (`int`, default: `200`),
                * 'alpha'         - smoothing factor (`float`, default: `0.8`).

    Examples
    --------
    Use the CEM optimizer `SCEM` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.cem.scem import SCEM
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 100,
       ...            'lower_boundary': -5 * numpy.ones((100,)),
       ...            'upper_boundary': 5 * numpy.ones((100,))}
       >>> options = {'max_function_evaluations': 1000000,  # set optimizer options
       ...            'n_individuals': 20,
       ...            'n_parents': 10,
       ...            'seed_rng': 2022,
       ...            'mean': 4 * np.ones((100,)),
       ...            'sigma': 0.1}
       >>> scem = SCEM(problem, options)  # initialize the optimizer class
       >>> results = scem.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"SCEM: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       SCEM: 1000000, 296947.9431526677

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring, offspring population size.
    n_parents     : `int`
                    number of parents, parental population size.
    mean          : `array_like`
                    mean of Gaussian search distribution.
    sigma         : `float`
                    mutation strength.
    alpha         : `float`
                    smoothing factor

    References
    ----------
    Kroese, D.P., Porotsky, S. and Rubinstein, R.Y., 2006.
    The cross-entropy method for continuous multi-extremal optimization.
    Methodology and Computing in Applied Probability, 8(3), pp.383-407.
    https://link.springer.com/article/10.1007/s11009-006-9753-0
    (See [B Main CE Program] for the official Matlab code.)

    De Boer, P.T., Kroese, D.P., Mannor, S. and Rubinstein, R.Y., 2005.
    A tutorial on the cross-entropy method.
    Annals of Operations Research, 134(1), pp.19-67.
    https://link.springer.com/article/10.1007/s10479-005-5724-z
    """
    def __init__(self, problem, options):
        CEM.__init__(self, problem, options)
        self.alpha = options.get('alpha', 0.8)  # smoothing factor

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
        self._sigmas = self.alpha*np.std(xx, axis=0) + (1.0-self.alpha)*self._sigmas
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
