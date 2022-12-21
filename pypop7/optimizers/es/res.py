import numpy as np

from pypop7.optimizers.es.es import ES


class RES(ES):
    """Rechenberg's (1+1)-Evolution Strategy with 1/5th success rule (RES).

    .. note:: `RES` is the *first* ES with self-adaptation of the *global* step-size, designed by Rechenberg, one
       recipient of `IEEE Evolutionary Computation Pioneer Award 2002 <https://tinyurl.com/456as566>`_. As
       **theoretically** investigated in Rechenberg's seminal PhD dissertation, the existence of the narrow **evolution
       window** eplains the necessarity of step-size adaptation to maximize local convergence progress.

       Since there is only one parent and only one offspring for each generation, `RES` generally shows very
       limited *exploration* ability for large-scale black-box optimization (LSBBO). Therefore, it is **highly
       recommended** to first attempt more advanced ES variants (e.g. `LMCMA`, `LMMAES`) for LSBBO. Here we
       include it mainly for *benchmarking* and *theoretical* purpose.

       AKA two-membered ES (which can also be seen as gradient climbing).

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
                * 'sigma'    - initial global step-size, aka mutation strength (`float`),
                * 'mean'     - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'lr_sigma' - learning rate of global step-size self-adaptation (`float`, default:
                  `1.0/np.sqrt(self.ndim_problem + 1.0)`).

    Examples
    --------
    Use the optimizer `RES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.res import RES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> res = RES(problem, options)  # initialize the optimizer class
       >>> results = res.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"RES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       RES: 5000, 0.06701744137207027

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/5n6ndrn7>`_ for more details.

    Attributes
    ----------
    lr_sigma : `float`
               learning rate of global step-size self-adaptation.
    mean     : `array_like`
               initial (starting) point, aka mean of Gaussian search distribution.
    sigma    : `float`
               final global step-size, aka mutation strength.

    References
    ----------
    Hansen, N., Arnold, D.V. and Auger, A., 2015.
    Evolution strategies.
    In Springer Handbook of Computational Intelligence (pp. 871-898). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007%2F978-3-662-43505-2_44
    (See Algorithm 44.3 for details.)

    Beyer, H.G. and Schwefel, H.P., 2002.
    Evolution strategies–A comprehensive introduction.
    Natural Computing, 1(1), pp.3-52.
    https://link.springer.com/article/10.1023/A:1015059928466

    Rechenberg, I., 1989.
    Evolution strategy: Nature’s way of optimization.
    In Optimization: Methods and Applications, Possibilities and Limitations (pp. 106-126).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-642-83814-9_6

    Rechenberg, I., 1984.
    The evolution strategy. A mathematical model of darwinian evolution.
    In Synergetics—from Microscopic to Macroscopic Order (pp. 122-132). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-642-69540-7_13
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        if self.lr_sigma is None:
            self.lr_sigma = 1.0/np.sqrt(self.ndim_problem + 1.0)
        assert self.lr_sigma > 0, f'`self.lr_sigma` = {self.lr_sigma}, but should > 0.'

    def initialize(self, args=None, is_restart=False):
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        y = self._evaluate_fitness(mean, args)  # fitness
        best_so_far_y = np.copy(y)
        return mean, y, best_so_far_y

    def iterate(self, args=None, mean=None):
        # sample and evaluate only one offspring
        x = mean + self.sigma*self.rng_optimization.standard_normal((self.ndim_problem,))
        y = self._evaluate_fitness(x, args)
        return x, y

    def restart_reinitialize(self, args=None, mean=None, y=None, best_so_far_y=None, fitness=None):
        self._fitness_list.append(y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (np.max(self._fitness_list[-self.stagnation:]) -
                            np.min(self._fitness_list[-self.stagnation:])) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self._print_verbose_info(fitness, y, True)
            self._n_restart += 1
            self._n_generations = 0
            self.sigma = np.copy(self._sigma_bak)
            mean, y, best_so_far_y = self.initialize(args, True)
            if self.saving_fitness:
                fitness.append(y)
            self._fitness_list = [best_so_far_y]
            if self.verbose:
                print(' ....... restart .......')
            self._print_verbose_info(fitness, y)
            self._n_generations = 1
        return mean, y, best_so_far_y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        mean, y, best_so_far_y = self.initialize(args)
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            x, y = self.iterate(args, mean)
            self._n_generations += 1
            self.sigma *= np.power(np.exp(float(y < best_so_far_y) - 0.2), self.lr_sigma)
            if y < best_so_far_y:
                mean, best_so_far_y = x, y
            if self.is_restart:
                mean, y, best_so_far_y = self.restart_reinitialize(
                    args, mean, y, best_so_far_y, fitness)
        return self._collect_results(fitness, mean, y)
