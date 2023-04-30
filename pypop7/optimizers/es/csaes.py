import numpy as np

from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.dsaes import DSAES


class CSAES(DSAES):
    """Cumulative Step-size self-Adaptation Evolution Strategy (CSAES).

    .. note:: `CSAES` adapts all the *individual* step-sizes on-the-fly with a *relatively small* population,
       according to the well-known `CSA <http://link.springer.com/chapter/10.1007/3-540-58484-6_263>`_ rule
       from the Evolutionary Computation community. The default setting (i.e., using a `small` population)
       may result in *relatively fast* (local) convergence, but with the risk of getting trapped in suboptima
       on multi-modal fitness landscape (which can be alleviated via *restart*).

       AKA cumulative (evolution) path-length control.

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

                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default:
                  `4 + int(np.floor(3*np.log(problem['ndim_problem'])))`),
                * 'n_parents'     - number of parents, aka parental population size (`int`, default:
                  `int(options['n_individuals']/4)`),
                * 'lr_sigma'      - learning rate of global step-size adaptation (`float`, default:
                  `np.sqrt(options['n_parents']/(problem['ndim_problem'] + options['n_parents']))`).

    Examples
    --------
    Use the optimizer `CSAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.csaes import CSAES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> csaes = CSAES(problem, options)  # initialize the optimizer class
       >>> results = csaes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CSAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CSAES: 5000, 0.010143683086819875

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/2s4ctvdw>`_ for more details.

    Attributes
    ----------
    lr_sigma      : `float`
                    learning rate of global step-size adaptation.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.
    sigma         : `float`
                    initial global step-size, aka mutation strength.

    References
    ----------
    Hansen, N., Arnold, D.V. and Auger, A., 2015.
    Evolution strategies.
    In Springer Handbook of Computational Intelligence (pp. 871-898). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007%2F978-3-662-43505-2_44

    Kern, S., Müller, S.D., Hansen, N., Büche, D., Ocenasek, J. and Koumoutsakos, P., 2004.
    Learning probability distributions in continuous evolutionary algorithms–a comparative review.
    Natural Computing, 3, pp.77-112.
    https://link.springer.com/article/10.1023/B:NACO.0000023416.59689.4e

    Ostermeier, A., Gawelczyk, A. and Hansen, N., 1994, October.
    Step-size adaptation based on non-local use of selection information
    In International Conference on Parallel Problem Solving from Nature (pp. 189-198).
    Springer, Berlin, Heidelberg.
    http://link.springer.com/chapter/10.1007/3-540-58484-6_263
    """
    def __init__(self, problem, options):
        if options.get('n_individuals') is None:  # number of offspring, aka offspring population size
            options['n_individuals'] = 4 + int(np.floor(3*np.log(problem['ndim_problem'])))
        if options.get('n_parents') is None:  # number of parents, aka parental population size
            options['n_parents'] = int(options['n_individuals']/4)
        if options.get('lr_sigma') is None:  # learning rate of global step-size adaptation
            options['lr_sigma'] = np.sqrt(options['n_parents']/(
                    problem['ndim_problem'] + options['n_parents']))
        DSAES.__init__(self, problem, options)
        self._s_1 = None
        self._s_2 = None
        # set E[||N(0,I)||]: expectation of chi distribution
        self._e_chi = np.sqrt(self.ndim_problem)*(
                1.0 - 1.0/(4.0*self.ndim_problem) + 1.0/(21.0*np.power(self.ndim_problem, 2)))

    def initialize(self, is_restart=False):
        self._s_1 = 1.0 - self.lr_sigma
        self._s_2 = np.sqrt(self.lr_sigma*(2.0 - self.lr_sigma)*self.n_parents)
        self._axis_sigmas = self._sigma_bak*np.ones((self.ndim_problem,))
        z = np.empty((self.n_individuals, self.ndim_problem))  # noise for offspring population
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        s = np.zeros((self.ndim_problem,))  # evolution path
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        self._list_initial_mean.append(np.copy(mean))
        return z, x, mean, s, y

    def iterate(self, z=None, x=None, mean=None, y=None, args=None):
        for k in range(self.n_individuals):  # to sample offspring population
            if self._check_terminations():
                return z, x, y
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            x[k] = mean + self._axis_sigmas*z[k]
            y[k] = self._evaluate_fitness(x[k], args)
        return z, x, y

    def _update_distribution(self, z=None, x=None, s=None, y=None):
        order = np.argsort(y)[:self.n_parents]
        s = self._s_1*s + self._s_2*np.mean(z[order], axis=0)
        sigmas_1 = np.power(np.exp(np.abs(s)/self._e_hnd - 1.0), 1.0/(3.0*self.ndim_problem))
        sigmas_2 = np.power(np.exp(np.linalg.norm(s)/self._e_chi - 1.0),
                            self.lr_sigma/(1.0 + np.sqrt(self.n_parents/self.ndim_problem)))
        self._axis_sigmas *= (sigmas_1*sigmas_2)
        mean = np.mean(x[order], axis=0)
        return s, mean

    def restart_reinitialize(self, z=None, x=None, mean=None, s=None, y=None):
        if not self.is_restart:
            return z, x, mean, s, y
        min_y = np.min(y)
        if min_y < self._list_fitness[-1]:
            self._list_fitness.append(min_y)
        else:
            self._list_fitness.append(self._list_fitness[-1])
        is_restart_1, is_restart_2 = np.all(self._axis_sigmas < self.sigma_threshold), False
        if len(self._list_fitness) >= self.stagnation:
            is_restart_2 = (self._list_fitness[-self.stagnation] - self._list_fitness[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self._print_verbose_info([], y, True)
            if self.verbose:
                print(' ....... *** restart *** .......')
            self._n_restart += 1
            self._list_generations.append(self._n_generations)  # for each restart
            self._n_generations = 0
            self.n_individuals *= 2
            self.n_parents = int(self.n_individuals/4)
            self.lr_sigma = np.sqrt(self.n_parents/(self.ndim_problem + self.n_parents))
            z, x, mean, s, y = self.initialize(True)
            self._list_fitness = [np.Inf]
        return z, x, mean, s, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        z, x, mean, s, y = self.initialize()
        while True:
            # sample and evaluate offspring population
            z, x, y = self.iterate(z, x, mean, y, args)
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            s, mean = self._update_distribution(z, x, s, y)
            z, x, mean, s, y = self.restart_reinitialize(z, x, mean, s, y)
        results = self._collect(fitness, y, mean)
        results['s'] = s
        results['_axis_sigmas'] = self._axis_sigmas
        return results
