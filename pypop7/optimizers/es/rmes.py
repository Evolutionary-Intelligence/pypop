import numpy as np

from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.r1es import R1ES


class RMES(R1ES):
    """Rank-M Evolution Strategy (RMES).

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
                * 'sigma'         - initial global step-size (σ), mutation strength (`float`),
                * 'mean'          - initial (starting) point, mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'n_evolution_paths' - number of evolution paths (`int`, default: `2`),
                * 'generation_gap'    - generation gap (`int`, default: `self.ndim_problem`),
                * 'n_individuals'     - number of offspring (λ: lambda), offspring population size (`int`, default:
                  `4 + int(3*np.log(self.ndim_problem))`),
                * 'n_parents'         - number of parents (μ: mu), parental population size (`int`, default:
                  `int(self.n_individuals/2)`),
                * 'c_cov'             - learning rate of low-rank covariance matrix (`float`, default:
                  `1.0/(3.0*np.sqrt(self.ndim_problem) + 5.0)`),
                * 'd_sigma'           - change factor of cumulative (global) step-size adaptation (`float`, default:
                  `1.0`).

    Examples
    --------
    Use the ES optimizer `RMES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.rmes import RMES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3 * numpy.ones((2,)),
       ...            'sigma': 0.1}
       >>> rmes = RMES(problem, options)  # initialize the optimizer class
       >>> results = rmes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"RMES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       RMES: 5000, 0.0005698039779344841

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/2aywpp2p>`_ for more details.

    Attributes
    ----------
    n_individuals     : `int`
                        number of offspring (λ: lambda), offspring population size.
    n_parents         : `int`
                        number of parents (μ: mu), parental population size.
    mean              : `array_like`
                        mean of Gaussian search distribution.
    sigma             : `float`
                        mutation strength.
    n_evolution_paths : `int`
                        number of evolution paths
    generation_gap    : `int`
                        generation gap.
    c_cov             : `float`
                        learning rate of low-rank covariance matrix.
    d_sigma           : `float`
                        change factor of cumulative (global) step-size adaptation.

    References
    ----------
    Li, Z. and Zhang, Q., 2018.
    A simple yet efficient evolution strategy for large-scale black-box optimization.
    IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
    https://ieeexplore.ieee.org/abstract/document/8080257
    """
    def __init__(self, problem, options):
        R1ES.__init__(self, problem, options)
        self.n_evolution_paths = options.get('n_evolution_paths', 2)  # m in Algorithm 2
        self.generation_gap = options.get('generation_gap', self.ndim_problem)  # T in Algorithm 2
        self._a = np.sqrt(1.0 - self.c_cov)  # for Line 4 in Algorithm 3
        self._a_m = np.power(self._a, self.n_evolution_paths)  # for Line 4 in Algorithm 3
        self._b = np.sqrt(self.c_cov)  # for Line 4 in Algorithm 3

    def initialize(self, args=None, is_restart=False):
        x, mean, p, s, y = R1ES.initialize(self, args, is_restart)
        mp = np.zeros((self.n_evolution_paths, self.ndim_problem))  # multiple evolution paths
        t_hat = np.zeros((self.n_evolution_paths,))  # for Algorithm 2
        return x, mean, p, s, mp, t_hat, y

    def iterate(self, x=None, mean=None, mp=None, y=None, args=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            z = self.rng_optimization.standard_normal((self.ndim_problem,))
            sum_p = np.zeros((self.ndim_problem,))
            for i in np.arange(self.n_evolution_paths) + 1:
                r = self.rng_optimization.standard_normal()
                sum_p += np.power(self._a, self.n_evolution_paths - i)*r*mp[i - 1]
            x[k] = mean + self.sigma*(self._a_m*z + self._b*sum_p)
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _update_distribution(self, x=None, mean=None, p=None, s=None,
                             mp=None, t_hat=None, y=None, y_bak=None):
        mean, p, s = R1ES._update_distribution(self, x, mean, p, s, y, y_bak)
        # update multiple evolution paths
        t_min = np.min(np.diff(t_hat))  # Line 2 in Algorithm 2 (T_min)
        i_apostrophe = np.argmin(np.diff(t_hat))  # Line 6 in Algorithm 2 (i')
        i_apostrophe += 1
        if (t_min > self.generation_gap) or (self._n_generations < self.n_evolution_paths):
            i_apostrophe = 0
        for i in range(i_apostrophe, self.n_evolution_paths - 1):
            mp[i], t_hat[i] = mp[i + 1], t_hat[i + 1]
        mp[-1], t_hat[-1] = p, self._n_generations
        return mean, p, s, mp, t_hat

    def restart_reinitialize(self, args=None, x=None, mean=None, p=None, s=None,
                             mp=None, t_hat=None, y=None, fitness=None):
        if ES.restart_reinitialize(self):
            x, mean, p, s, mp, t_hat, y = self.initialize(args, True)
            if self.saving_fitness:
                fitness.append(y[0])
            self.d_sigma *= 2.0
            self._print_verbose_info(y)
        return x, mean, p, s, mp, t_hat, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p, s, mp, t_hat, y = self.initialize(args)
        if self.saving_fitness:
            fitness.append(y[0])
        while True:
            y_bak = np.copy(y)
            # sample and evaluate offspring population
            x, y = self.iterate(x, mean, mp, y, args)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, p, s, mp, t_hat = self._update_distribution(x, mean, p, s, mp, t_hat, y, y_bak)
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                x, mean, p, s, mp, t_hat, y = self.restart_reinitialize(
                    args, x, mean, p, s, mp, t_hat, y, fitness)
        results = self._collect_results(fitness, mean)
        results['p'] = p
        results['s'] = s
        return results
