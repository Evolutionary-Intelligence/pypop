from pypop7.optimizers.rs.prs import PRS


class RHC(PRS):
    """Random (stochastic) Hill Climber (RHC).

    .. note:: Currently `RHC` only supports *normally*-distributed random sampling during optimization.
       It often suffers from **slow convergence** for large-scale black-box optimization (LSBBO), owing to
       its limited exploration ability originating from its individual-based local sampling strategy.
       Therefore, it is **highly recommended** to first attempt other more advanced (e.g. population-based)
       methods for LSBBO.

       "They have two key advantages: (1) they use very little memory; and (2) they can often find reasonable solutions
       in large or infinite state spaces for which systematic algorithms are unsuitable."---`[Russell&Norvig, 2021]
       <http://aima.cs.berkeley.edu/>`_

       **"The hill-climbing search algorithm is the most basic local search technique."**---`[Russell&Norvig, 2021]
       <http://aima.cs.berkeley.edu/>`_

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
                * 'x'                           - initial (starting) point (`array_like`),
                * 'sigma'                       - initial global step-size (`float`),
                * 'initialization_distribution' - random sampling distribution for starting point initialization (`int`,
                  default: `1`). Only when `x` is not set *explicitly*, it will be used.

                  * `1`: *uniform* distribution is used for random sampling,
                  * `0`: *standard normal* distribution is used for random sampling with mean `0` and std `1` for
                    each dimension.

    Examples
    --------
    Use the optimizer `RHC` to minimize the well-known test function
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
       >>> print(f"RHC: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       RHC: 5000, 7.13722829962456e-05

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/3u864ju3>`_ for more details.

    Attributes
    ----------
    initialization_distribution : `int`
                                  random sampling distribution for initialization of starting point.
    sigma                       : `float`
                                  initial global step-size.
    x                           : `array_like`
                                  initial (starting) point.

    References
    ----------
    Russell, S. and Norvig P., 2021.
    Artificial intelligence: A modern approach (Global Edition).
    Pearson Education.
    http://aima.cs.berkeley.edu/    (See CHAPTER 4: SEARCH IN COMPLEX ENVIRONMENTS)

    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/hillclimber.py
    """
    def __init__(self, problem, options):
        # only support normally-distributed random sampling during optimization
        options['sampling_distribution'] = 0
        PRS.__init__(self, problem, options)
        # set default: 1 -> uniformly distributed random sampling
        self.initialization_distribution = options.get('initialization_distribution', 1)
        if self.initialization_distribution not in [0, 1]:  # 0 -> normally distributed random sampling
            info = 'For {:s}, only support uniformly or normally distributed random initialization.'
            raise ValueError(info.format(self.__class__.__name__))

    def _sample(self, rng):  # only for `initialize(self)`
        if self.initialization_distribution == 0:  # normally distributed
            x = rng.standard_normal(size=(self.ndim_problem,))
        else:  # uniformly distributed
            x = rng.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        return x

    def iterate(self):  # sampling via mutating the best-so-far individual
        noise = self.rng_optimization.standard_normal(size=(self.ndim_problem,))
        return self.best_so_far_x + self.sigma*noise
