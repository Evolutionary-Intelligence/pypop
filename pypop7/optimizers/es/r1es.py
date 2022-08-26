import numpy as np

from pypop7.optimizers.es.es import ES


class R1ES(ES):
    """Rank-One Evolution Strategy (R1ES).

    .. note:: `R1ES` is a **low-rank** version of `CMA-ES` specifically designed for large-scale black-box optimization
       (LSBBO) by Li and `Zhang <https://tinyurl.com/32hsbx28>`_. It often works well when there is a dominated search
       direction embedded in the subspace. For more complex landscapes (e.g., there are multiple promising search
       directions), other competitive LSBBO variants (e.g., `Rm-ES`, `LM-CMA`, `LM-MA-ES`) may be more preferred.

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
                * 'verbose_frequency'        - generation frequency of printing verbose info (`int`, default: `10`);
              and with four particular settings (`keys`):
                * 'sigma'         - initial global step-size (σ), mutation strength (`float`),
                * 'mean'          - initial (starting) point, mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`).

                * 'n_individuals' - number of offspring (λ: lambda), offspring population size (`int`, default:
                  `4 + int(3*np.log(self.ndim_problem))`),
                * 'n_parents'     - number of parents (μ: mu), parental population size (`int`, default:
                  `int(self.n_individuals / 2)`),
                * 'c_cov'         - learning rate of low-rank covariance matrix (`float`, default:
                  `1.0/(3.0*np.sqrt(self.ndim_problem) + 5.0)`),
                * 'c'             - learning rate of evolution path (`float`, default: `2.0/(self.ndim_problem + 7.0)`),
                * 'c_s'           - learning rate of cumulative (global) step-size adaptation (`float`, default: `0.3`),
                * 'q_star'        - baseline of cumulative (global) step-size adaptation (`float`, default: `0.3`)，
                * 'd_sigma'       - change factor of cumulative (global) step-size adaptation (`float`, default: `1.0`).

    Examples
    --------
    Use the ES optimizer `R1ES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.r1es import R1ES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3 * numpy.ones((2,)),
       ...            'sigma': 0.1}
       >>> r1es = R1ES(problem, options)  # initialize the optimizer class
       >>> results = r1es.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"R1ES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       R1ES: 5000, 0.0005057573524421161

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/2aywpp2p>`_ for more details.

    Attributes
    ----------
    n_individuals   : `int`
                      number of offspring (λ: lambda), offspring population size.
    n_parents       : `int`
                      number of parents (μ: mu), parental population size.
    mean            : `array_like`
                      initial (starting) point, mean of Gaussian search distribution.
    sigma           : `float`
                      initial global step-size (σ), mutation strength (`float`).
    c_cov           : `float`
                      learning rate of low-rank covariance matrix.
    c               : `float`
                      learning rate of evolution path.
    c_s             : `float`
                      learning rate of cumulative (global) step-size adaptation.
    q_star          : `float`
                      baseline of cumulative (global) step-size adaptation.
    d_sigma         : `float`
                      change factor of cumulative (global) step-size adaptation.

    References
    ----------
    Li, Z. and Zhang, Q., 2018.
    A simple yet efficient evolution strategy for large-scale black-box optimization.
    IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
    https://ieeexplore.ieee.org/abstract/document/8080257
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.c_cov = options.get('c_cov', 1.0/(3.0*np.sqrt(self.ndim_problem) + 5.0))  # for Line 5 in Algorithm 1
        self.c = options.get('c', 2.0/(self.ndim_problem + 7.0))  # for Line 12 in Algorithm 1 (c_c)
        self.c_s = options.get('c_s', 0.3)  # for Line 15 in Algorithm 1
        self.q_star = options.get('q_star', 0.3)  # for Line 15 in Algorithm 1
        self.d_sigma = options.get('d_sigma', 1.0)  # for Line 16 in Algorithm 1
        self._x_1 = np.sqrt(1.0 - self.c_cov)  # for Line 5 in Algorithm 1
        self._x_2 = np.sqrt(self.c_cov)  # for Line 5 in Algorithm 1
        self._p_1 = 1.0 - self.c  # for Line 12 in Algorithm 1
        self._p_2 = None  # for Line 12 in Algorithm 1
        self._rr = None  # for rank-based success rule (RSR)

    def initialize(self, args=None, is_restart=False):
        self._p_2 = np.sqrt(self.c*(2.0 - self.c)*self._mu_eff)
        self._rr = np.arange(self.n_parents*2) + 1
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        p = np.zeros((self.ndim_problem,))  # principal search direction
        s = 0.0  # cumulative rank rate
        y = np.tile(self._evaluate_fitness(mean, args), (self.n_individuals,))  # fitness
        return x, mean, p, s, y

    def iterate(self, x=None, mean=None, p=None, y=None, args=None):
        for k in range(self.n_individuals):  # for Line 3 in Algorithm 1
            if self._check_terminations():
                return x, y
            z = self.rng_optimization.standard_normal((self.ndim_problem,))  # for Line 4 in Algorithm 1
            r = self.rng_optimization.standard_normal()  # for Line 4 in Algorithm 1
            # for Line 5 in Algorithm 1
            x[k] = mean + self.sigma*(self._x_1*z + self._x_2*r*p)
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _update_distribution(self, x=None, mean=None, p=None, s=None, y=None, y_bak=None):
        order = np.argsort(y)
        y.sort()  # for Line 10 in Algorithm 1
        # for Line 11 in Algorithm 1
        mean_w = np.zeros((self.ndim_problem,))
        for k in range(self.n_parents):
            mean_w += self._w[k]*x[order[k]]
        p = self._p_1*p + self._p_2*(mean_w - mean) / self.sigma  # for Line 12 in Algorithm 1
        mean = mean_w  # for Line 11 in Algorithm 1
        # for rank-based adaptation of mutation strength
        r = np.argsort(np.hstack((y_bak[:self.n_parents], y[:self.n_parents])))
        rr = self._rr[r < self.n_parents] - self._rr[r >= self.n_parents]
        q = np.dot(self._w, rr) / self.n_parents  # for Line 14 in Algorithm 1
        s = (1 - self.c_s)*s + self.c_s*(q - self.q_star)  # for Line 15 in Algorithm 1
        self.sigma *= np.exp(s / self.d_sigma)  # for Line 16 in Algorithm 1
        return mean, p, s

    def restart_initialize(self, args=None, x=None, mean=None, p=None, s=None, y=None, fitness=None):
        is_restart = ES.restart_initialize(self)
        if is_restart:
            x, mean, p, s, y = self.initialize(args, is_restart)
            fitness.append(y[0])
            self.d_sigma *= 2.0
        return x, mean, p, s, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p, s, y = self.initialize(args)
        fitness.append(y[0])
        while True:
            y_bak = np.copy(y)  # for Line 13 in Algorithm 1
            x, y = self.iterate(x, mean, p, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, p, s = self._update_distribution(x, mean, p, s, y, y_bak)
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                x, mean, p, s, y = self.restart_initialize(args, x, mean, p, s, y, fitness)
        results = self._collect_results(fitness, mean)
        results['p'] = p
        results['s'] = s
        return results
