from pypop7.optimizers.rs.prs import PRS


class RHC(PRS):
    """Random Hill Climber (RHC).

    .. note:: Currently `RHC` only supports *normally*-distributed random sampling during optimization.
       But it supports two ways of random sampling (*uniformly* or *normally* distributed) for the initial
       (starting) point. It often suffers from **very slow convergence** for large-scale black-box optimization
       (LSBBO), since its limited exploration ability originating from its individual-based sampling strategy.

       It is **highly recommended** to first attempt other more advanced methods for LSBBO.

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
                * 'verbose_frequency'        - frequency of printing verbose info (`int`, default: `1000`);
              and with three particular settings (`keys`):
                * 'x'                         - initial (starting) point (`array_like`),
                * 'sigma'                     - initial (global) step-size (`float`),
                * initialization_distribution - random sampling distribution for starting point initialization (`int`,
                  default: `1`).

                  * `1`: *uniform* distribution is used for random sampling,
                  * `0`: *standard normal* distribution is used for random sampling.

    Examples
    --------
    Use the Random Search optimizer `RHC` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.rs.rhc import RHC
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3 * numpy.ones((2,)),
       ...            'sigma': 0.1}
       >>> rhc = RHC(problem, options)  # initialize the optimizer class
       >>> results = rhc.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"Random-Hill-Climber: {results['n_function_evaluations']}, {results['best_so_far_y']}")
         * Generation 0: best_so_far_y 3.60400e+03, min(y) 3.60400e+03 & Evaluations 1
         * Generation 1000: best_so_far_y 2.00027e-01, min(y) 4.73637e+00 & Evaluations 1001
         * Generation 2000: best_so_far_y 8.02903e-04, min(y) 2.57241e-01 & Evaluations 2001
         * Generation 3000: best_so_far_y 7.13723e-05, min(y) 1.78250e+00 & Evaluations 3001
         * Generation 4000: best_so_far_y 7.13723e-05, min(y) 7.80693e+00 & Evaluations 4001
       Random-Hill-Climber: 5000, 7.13722829962456e-05

    Attributes
    ----------
    x                           : `array_like`
                                  initial (starting) point.
    sigma                       : `float`
                                  (global) step-size.
    initialization_distribution : `int`
                                  random sampling distribution for initialization of starting point.

    References
    ----------
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/hillclimber.py
    """
    def __init__(self, problem, options):
        # only support normally-distributed random sampling during optimization
        options['sampling_distribution'] = 0
        PRS.__init__(self, problem, options)
        # default: 1 -> uniformly distributed random sampling
        self.initialization_distribution = options.get('initialization_distribution', 1)
        if self.initialization_distribution not in [0, 1]:  # 0 -> normally distributed random sampling
            info = 'For {:s}, only support uniformly or normally distributed random initialization.'
            raise ValueError(info.format(self.__class__.__name__))

    def _sample(self, rng):  # only for `initialize(self)`, not for `iterate(self)`
        if self.initialization_distribution == 0:  # normally distributed
            x = rng.standard_normal(size=(self.ndim_problem,))
        else:  # uniformly distributed
            x = rng.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        return x

    def iterate(self):  # draw a sample via mutating the best-so-far individual
        noise = self.rng_optimization.standard_normal(size=(self.ndim_problem,))
        return self.best_so_far_x + self.sigma*noise
