from pypop7.optimizers.es.maes import MAES  # Matrix Adaptation Evolution Strategy


class FMAES(MAES):
    """Fast Matrix Adaptation Evolution Strategy (FMAES).

    .. note:: `FMAES` is a *more efficient* implementation of `MAES` with *quadractic* time complexity w.r.t. each
       sampling, which replaces the computationally expensive matrix-matrix multiplication (*cubic time complexity*)
       with the combination of matrix-matrix addition and matrix-vector multiplication (*quadractic time complexity*)
       for transformation matrix adaptation. It is **highly recommended** to first attempt more advanced `ES` variants
       (e.g., `LMCMA`, `LMMAES`) for large-scale black-box optimization, since `FMAES` still has a computationally
       intensive *quadratic* time complexity.

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
                * 'sigma'         - initial global step-size, aka mutation strength (`float`),
                * 'mean'          - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default:
                  `4 + int(3*np.log(problem['ndim_problem']))`),
                * 'n_parents'     - number of parents, aka parental population size (`int`, default:
                  `int(options['n_individuals']/2)`).

    Examples
    --------
    Use the black-box optimizer `FMAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.fmaes import FMAES
       >>> problem = {'fitness_function': rosenbrock,  # to define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5.0*numpy.ones((2,)),
       ...            'upper_boundary': 5.0*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # to set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3.0*numpy.ones((2,)),
       ...            'sigma': 3.0}  # global step-size may need to be fine-tuned for better performance
       >>> fmaes = FMAES(problem, options)  # to initialize the optimizer class
       >>> results = fmaes.optimize()  # to run the optimization/evolution process
       >>> print(f"FMAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       FMAES: 5000, 1.3259e-17

    For its correctness checking of Python coding, please refer to `this code-based repeatability report
    <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_fmaes.py>`_
    for all details. For *pytest*-based automatic testing, please see `test_fmaes.py
    <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/test_fmaes.py>`_.

    Attributes
    ----------
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.
    sigma         : `float`
                    final global step-size, aka mutation strength.

    References
    ----------
    Beyer, H.G., 2020, July.
    `Design principles for matrix adaptation evolution strategies.
    <https://dl.acm.org/doi/abs/10.1145/3377929.3389870>`_
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation Companion (pp. 682-700).

    Loshchilov, I., Glasmachers, T. and Beyer, H.G., 2019.
    `Large scale black-box optimization by limited-memory matrix adaptation.
    <https://ieeexplore.ieee.org/abstract/document/8410043>`_
    IEEE Transactions on Evolutionary Computation, 23(2), pp.353-358.

    Beyer, H.G. and Sendhoff, B., 2017.
    `Simplify your covariance matrix adaptation evolution strategy.
    <https://ieeexplore.ieee.org/document/7875115>`_
    IEEE Transactions on Evolutionary Computation, 21(5), pp.746-759.

    Please refer to the *official* Matlab version from Prof. Beyer:
    https://homepages.fhv.at/hgb/downloads/ForDistributionFastMAES.tar
    """
    def __init__(self, problem, options):
        options['_fast_version'] = True  # mandatory setting for only FMAES
        MAES.__init__(self, problem, options)
