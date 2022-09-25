import numpy as np

from pypop7.optimizers.es.es import ES


class SEPCMAES(ES):
    """Separable Covariance Matrix Adaptation Evolution Strategy (SEPCMAES).

    .. note:: `SEPCMAES` learns only the **diagonal** elements of the full covariance matrix explicitly, leading
       to a *linear* time complexity w.r.t. each sampling for large-scale black-box optimization.

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
                * 'mean'          - initial (starting) point, mean of Gaussian search distribution (`array_like`),
                * 'sigma'         - initial global step-size, mutation strength (`float`),
                * 'n_individuals' - number of offspring, offspring population size (`int`default:
                  `4 + int(3*np.log(self.ndim_problem))`),
                * 'n_parents'     - number of parents, parental population size (`int`, default:
                  `int(self.n_individuals / 2)`),
                * 'c_c'           - learning rate of evolution path update (`float`, default:
                  `4.0/(self.ndim_problem + 4.0)`).

    Examples
    --------
    Use the ES optimizer `SEPCMAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.sepcmaes import SEPCMAES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3 * numpy.ones((2,)),
       ...            'sigma': 0.1}
       >>> sepcmaes = SEPCMAES(problem, options)  # initialize the optimizer class
       >>> results = sepcmaes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"SEPCMAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       SEPCMAES: 5000, 0.010606023368122535

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/mpjzv8yh>`_ for more details.

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring/descendants (λ: lambda), offspring population size.
    n_parents     : `int`
                    number of parents (μ: mu), parental population size.
    mean          : `array_like`
                    initial (starting) point, mean of Gaussian search distribution.
    sigma         : `float`
                    initial global step-size (σ), mutation strength (rate).
    c_c           : `float`
                    learning rate of evolution path update.

    References
    ----------
    Ros, R. and Hansen, N., 2008, September.
    A simple modification in CMA-ES achieving linear time and space complexity.
    In International Conference on Parallel Problem Solving from Nature (pp. 296-305).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-540-87700-4_30
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.c_c = options.get('c_c', 4.0/(self.ndim_problem + 4.0))
        self.options = options
        self.c_s = None
        self.c_cov = None
        self.d_sigma = None
        self._s_1 = None
        self._s_2 = None

    def _set_c_cov(self):
        c_cov = (1.0/self._mu_eff)*(2.0/np.power(self.ndim_problem + np.sqrt(2.0), 2)) + (
            (1.0 - 1.0/self._mu_eff)*np.minimum(1.0, (2.0*self._mu_eff - 1.0) / (
                np.power(self.ndim_problem + 2.0, 2) + self._mu_eff)))
        c_cov *= (self.ndim_problem + 2.0)/3.0  # for faster adaptation
        return c_cov

    def _set_d_sigma(self):
        d_sigma = np.maximum((self._mu_eff - 1.0)/(self.ndim_problem + 1.0) - 1.0, 0.0)
        return 1.0 + self.c_s + 2.0*np.sqrt(d_sigma)

    def initialize(self, is_restart=False):
        self.c_s = self.options.get('c_s', (self._mu_eff + 2.0)/(self.ndim_problem + self._mu_eff + 3.0))
        self.c_cov = self.options.get('c_cov', self._set_c_cov())
        self.d_sigma = self.options.get('d_sigma', self._set_d_sigma())
        self._s_1 = 1.0 - self.c_s
        self._s_2 = np.sqrt(self._mu_eff*self.c_s*(2.0 - self.c_s))
        z = np.empty((self.n_individuals, self.ndim_problem))  # Gaussian noise for mutation
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        s = np.zeros((self.ndim_problem,))  # evolution path for CSA
        p = np.zeros((self.ndim_problem,))  # evolution path for CMA
        c = np.ones((self.ndim_problem,))  # diagonal elements for covariance matrix
        d = np.ones((self.ndim_problem,))  # diagonal elements for covariance matrix
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return z, x, mean, s, p, c, d, y

    def iterate(self, z=None, x=None, mean=None, d=None, y=None, args=None):
        for k in range(self.n_individuals):  # Line 2 of Algorithm 1
            if self._check_terminations():
                return z, x, y
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            x[k] = mean + self.sigma*d*z[k]  # Line 3 of Algorithm 1
            y[k] = self._evaluate_fitness(x[k], args)
        return z, x, y

    def _update_distribution(self, z=None, x=None, s=None, p=None, c=None, d=None, y=None):
        order = np.argsort(y)
        zeros = np.zeros((self.ndim_problem,))
        z_w, mean, dz_w = np.copy(zeros), np.copy(zeros), np.copy(zeros)
        for k in range(self.n_parents):  # Line 4, 5 of Algorithm 1
            z_w += self._w[k]*z[order[k]]
            mean += self._w[k]*x[order[k]]  # update distribution mean
            dz = d*z[order[k]]
            dz_w += self._w[k]*dz*dz
        s = self._s_1*s + self._s_2*z_w  # Line 6 of Algorithm 1
        # Line 7 of Algorithm 1
        if (np.linalg.norm(s) / np.sqrt(1.0 - np.power(1.0 - self.c_s, 2.0*self._n_generations))) < (
                (1.4 + 2.0/(self.ndim_problem + 1.0))*self._e_chi):
            h = np.sqrt(self.c_c*(2.0 - self.c_c))*np.sqrt(self._mu_eff)*d*z_w
        else:
            h = 0
        p = (1.0 - self.c_c)*p + h  # Line 8 of Algorithm 1
        c = (1.0 - self.c_cov)*c + (1.0/self._mu_eff)*self.c_cov*p*p + (
                self.c_cov*(1.0 - 1.0/self._mu_eff)*dz_w)  # Line 9 of Algorithm 1
        # implement Line 10 of Algorithm 1
        self.sigma *= np.exp(self.c_s/self.d_sigma*(np.linalg.norm(s)/self._e_chi - 1.0))
        d = np.sqrt(c)  # Line 11 of Algorithm 1
        return mean, s, p, c, d

    def restart_reinitialize(self, z=None, x=None, mean=None, s=None, p=None, c=None, d=None, y=None):
        is_restart = ES.restart_reinitialize(self)
        if is_restart:
            z, x, mean, s, p, c, d, y = self.initialize(is_restart)
        return z, x, mean, s, p, c, d, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        z, x, mean, s, p, c, d, y = self.initialize()
        while True:
            self._n_generations += 1  # Line 1 of Algorithm 1
            # sample and evaluate offspring population
            z, x, y = self.iterate(z, x, mean, d, y, args)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, s, p, c, d = self._update_distribution(z, x, s, p, c, d, y)
            self._print_verbose_info(y)
            if self.is_restart:
                z, x, mean, s, p, c, d, y = self.restart_reinitialize(z, x, mean, s, p, c, d, y)
        results = self._collect_results(fitness, mean)
        results['s'] = s
        results['p'] = p
        results['d'] = d
        return results
