import numpy as np

from pypop7.optimizers.cem.cem import CEM
import colorednoise


class SECEM(CEM):
    """Sample-efficient Cross-Entropy Method

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
              and with the following particular settings (`keys`):
                * 'n_individuals' - population size (`int`, default: `1000`),
                * 'n_parents' - parent size (`int`, default: `200`),
                * 'mean' - initial mean value (`array_like`, default: `4 * np.ones((ndim_problem,))`),
                * 'sigma' - initial global step-size (σ), mutation strength (`float`, default: '1.0'),
                * 'fraction_reused_elites' - reuse fraction (`float`, default: `0.3`),
                * 'noise_beta' - noise parameter (`float`, default: `2.0`),

    Examples
    --------
    Use the CEM optimizer `SECEM` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.cem.secem import SECEM
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 100,
       ...            'lower_boundary': -5 * numpy.ones((100,)),
       ...            'upper_boundary': 5 * numpy.ones((100,))}
       >>> options = {'max_function_evaluations': 1000000,  # set optimizer options
       ...            'n_individuals': 100,
       ...            'n_parents': 20,
       ...            'mean': 4 * np.ones((100,)),
       ...            'fraction_reused_elites': 0.3,
       ...            'sigma': 0.5,
       ...            'noise_beta': 2.0,
       ...            'seed_rng': 2022}
       >>> secem = SECEM(problem, options)  # initialize the optimizer class
       >>> results = secem.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"SECEM: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       SECEM: 1000000, 1.7065180618298407e-05

    Attributes
    ----------
    n_individuals           : `int`
                              number of offspring (λ: lambda), offspring population size.
    n_parents               : `int`
                              number of parents (μ: mu), parental population size.
    mean                    : `array_like`
                              mean of Gaussian search distribution.
    sigma                   : `float`
                              mutation strength.
    fraction_reused_elites  : `float`
                              reuse fraction
    noise_beta              : `float`
                              noise parameter

    Reference
    ---------------
    C. Pinneri, S. Sawant, S. Blaes, J. Achterhold, J. Stuckler, M. Rolinek, G. Martius
    Sample-efficient Cross-Entropy Method for Real-time Planning
    https://arxiv.org/pdf/2008.06389.pdf
    """
    def __init__(self, problem, options):
        CEM.__init__(self, problem, options)
        self.alpha = 0.1
        self.fraction_reused_elites = options.get('fraction_reused_elites', 0.3)
        self.decay = 1.25  # decay parameter used in N
        self.noise_beta = options.get('noise_beta', 2.0)

    def initialize(self, is_restart=False):
        mean = self._initialize_mean(is_restart)
        y = np.empty((self.n_individuals,))
        x = np.empty((self.n_individuals, self.ndim_problem))
        elite_x = np.empty((self.n_parents, self.ndim_problem))
        elite_y = np.empty((self.n_parents,))
        return mean, x, elite_x, y, elite_y

    def re_initialize(self):
        y = np.empty((self.n_individuals,))
        return y

    def iterate(self, mean, x, y):
        for i in range(self.n_individuals):
            x[i] = colorednoise.powerlaw_psd_gaussian(self.noise_beta, size=(1, self.ndim_problem))
            x[i] = np.clip(np.dot(x[i], self.sigma) + self.mean, self.lower_boundary, self.upper_boundary)
            y[i] = self._evaluate_fitness(x[i])
            if self._check_terminations():
                return x, y
        return x, y

    def _update_parameters(self, mean, x, elite_x, y, elite_y):
        if self._n_generations == 0:
            order = np.argsort(y)
            for j in range(self.n_parents):
                elite_x[j] = x[order[j]]
                elite_y[j] = y[order[j]]
        else:
            length = int(self.fraction_reused_elites * self.n_parents)
            combine_x = np.empty((length, self.ndim_problem))
            combine_y = np.empty((length,))
            for k in range(length):
                combine_x[k] = elite_x[k]
                combine_y[k] = elite_y[k]
            combine_x = np.vstack((x, combine_x))
            combine_y = np.hstack((y, combine_y))
            order = np.argsort(combine_y)
            for j in range(self.n_parents):
                elite_x[j] = combine_x[order[j]]
                elite_y[j] = combine_y[order[j]]
        mean = self.alpha * mean + (1 - self.alpha) * np.mean(elite_x, axis=0)
        self.sigma = self.alpha * self.sigma + (1 - self.alpha) * np.std(elite_x, axis=0)
        return mean, elite_x, elite_y

    def optimize(self, fitness_function=None, args=None):
        fitness = CEM.optimize(self, fitness_function)
        mean, x, elite_x, y, elite_y = self.initialize()
        restart = True
        while True:
            self.n_individuals = int(max(self.n_individuals * np.power(self.decay, -1 * self._n_generations),
                                     2 * self.n_parents))
            if self.n_individuals != 2 * self.n_parents or restart is True:
                y = self.re_initialize()
                if self.n_individuals == 2 * self.n_parents:
                    restart = False
            x, y = self.iterate(mean, x, y)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, elite_x, elite_y = self._update_parameters(mean, x, elite_x, y, elite_y)
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness, mean)
        return results
