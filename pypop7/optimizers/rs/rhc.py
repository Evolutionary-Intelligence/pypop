from pypop7.optimizers.rs.prs import PRS


class RHC(PRS):
    """Random (stochastic) Hill Climber (RHC).

    .. note:: Currently `RHC` only supports normally-distributed random sampling during optimization. It often suffers
       from **slow convergence** for large-scale black-box optimization (LSBBO), owing to its *relatively limited*
       exploration ability originating from its **individual-based** sampling strategy. Therefore, it is **highly
       recommended** to first attempt more advanced (e.g. population-based) methods for LSBBO.

       `"The hill-climbing search algorithm is the most basic local search technique. They have two key advantages:
       (1) they use very little memory; and (2) they can often find reasonable solutions in large or infinite state
       spaces for which systematic algorithms are unsuitable."---[Russell&Norvig, 2021]
       <http://aima.cs.berkeley.edu/>`_

       AKA `"stochastic local search (steepest ascent or greedy search)"---[Murphy., 2022]
       <https://probml.github.io/pml-book/book2.html>`_.

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
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'sigma'             - initial global step-size (`float`),
                * 'x'                 - initial (starting) point (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`, when `init_distribution`
                    is `1`. Otherwise, *standard normal* distributed random sampling is used.

                * 'init_distribution' - random sampling distribution for starting-point initialization (`int`,
                  default: `1`). Only when `x` is not set *explicitly*, it will be used.

                  * `1`: *uniform* distributed random sampling only for starting-point initialization,
                  * `0`: *standard normal* distributed random sampling only for starting-point initialization.

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.rs.rhc import RHC
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}
       >>> rhc = RHC(problem, options)  # initialize the optimizer class
       >>> results = rhc.optimize()  # run the optimization process
       >>> # return the number of used function evaluations and found best-so-far fitness
       >>> print(f"RHC: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       RHC: 5000, 7.13722829962456e-05

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/3u864ju3>`_ for more details.

    Attributes
    ----------
    init_distribution : `int`
                        random sampling distribution for starting-point initialization.
    sigma             : `float`
                        global step-size (fixed during optimization).
    x                 : `array_like`
                        initial (starting) point.

    References
    ----------
    The following code from PyBrain directly inspired the coding of `RHC`:
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/hillclimber.py

    For the following book, Chapter 6.7 (DFO) gives an introduction of `RHC`:
    https://probml.github.io/pml-book/book2.html

    For the following book, Chapter 4 (SEARCH IN COMPLEX ENVIRONMENTS) gives an introduction of `RHC`:
    Russell, S. and Norvig P., 2021.
    `Artificial intelligence: A modern approach (Global Edition).
    <http://aima.cs.berkeley.edu/>`_
    Pearson Education.

    Hoos, H.H. and Stützle, T., 2004.
    `Stochastic local search: Foundations and applications.
    <https://www.elsevier.com/books/stochastic-local-search/hoos/978-1-55860-872-6>`_
    Elsevier.

    Baluja, S., 1996.
    `Genetic algorithms and explicit search statistics.
    <https://proceedings.neurips.cc/paper/1996/hash/e6d8545daa42d5ced125a4bf747b3688-Abstract.html>`_
    In Advances in Neural Information Processing Systems (pp.319-325).

    Juels, A. and Wattenberg, M., 1995.
    `Stochastic hillclimbing as a baseline method for evaluating genetic algorithms.
    <https://proceedings.neurips.cc/paper/1995/hash/36a1694bce9815b7e38a9dad05ad42e0-Abstract.html>`_
    In Advances in Neural Information Processing Systems (pp. 430-436).
    """
    def __init__(self, problem, options):
        # only support normally-distributed random sampling during optimization
        options['_sampling_type'] = 0  # 0 -> normally distributed random sampling (a mandatory setting)
        PRS.__init__(self, problem, options)
        # set default: 1 -> uniformly distributed random sampling
        self.init_distribution = options.get('init_distribution', 1)
        if self.init_distribution not in [0, 1]:  # 0 -> normally distributed random sampling
            info = 'For currently {:s}, only support uniformly or normally distributed random initialization.'
            raise ValueError(info.format(self.__class__.__name__))

    def _sample(self, rng):  # only for function `initialize(self)` inherited from the parent class `PRS`
        if self.init_distribution == 0:  # normally distributed
            x = rng.standard_normal(size=(self.ndim_problem,))
        else:  # uniformly distributed
            x = rng.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        return x

    def iterate(self):  # sampling via mutating the best-so-far individual
        noise = self.rng_optimization.standard_normal(size=(self.ndim_problem,))
        return self.best_so_far_x + self.sigma*noise  # mutation based on Gaussian-noise perturbation
