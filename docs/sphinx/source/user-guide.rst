User Guide
==========

Before applying `pypop7` to real-world black-box optimization problems, the following user guidelines should
be read carefully: *problem definition*, *optimizer setting*, *result analyses*, and *algorithm selection*.

Problem Definition
------------------

First, an *objective function* (called *fitness function* in this library) needs to be defined in the `function
<https://docs.python.org/3/reference/compound_stmts.html#function-definitions>`_ form. Then, for simplicity, the
data structure `dict <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`_ is used as an effective
way to store all settings related to the optimization problem at hand, such as:
  * `fitness_function`: objective function to be **minimized** (`func`),
  * `ndim_problem`: number of dimensionality (`int`),
  * `upper_boundary`: upper boundary of search range (`array_like`),
  * `lower_boundary`: lower boundary of search range (`array_like`),
  * `initial_upper_boundary`: upper boundary only for initialization (`array_like`),
  * `initial_lower_boundary`: lower boundary only for initialization (`array_like`).

Note that without loss of generality, only the **minimization** process is considered in this library, since
*maximization* can be easily transferred to *minimization* by negating it.

Both `initial_upper_boundary` and `initial_lower_boundary` are set to `upper_boundary` and `lower_boundary`,
respectively, if *not* given. When `initial_upper_boundary` and `initial_lower_boundary` are explicitly given,
the initialization of population/individual will be sampled from [`initial_lower_boundary`, `initial_upper_boundary`]
rather than [`lower_boundary`, `upper_boundary`]. This is *mainly* used for optimizers benchmarking purpose (in
order to avoid utilizing `symmetry and origin <https://www.tandfonline.com/doi/full/10.1080/10556788.2020.1808977>`_
to possibly bias the search).

Below is a simple example to define the well-known test function `Rosenbrock
<http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy as np
       >>> def rosenbrock(x):  # define the fitness (cost/objective) function
       ...     return 100.0*np.sum(np.power(x[1:] - np.power(x[:-1], 2), 2)) + np.sum(np.power(x[:-1] - 1, 2))
       >>> ndim_problem = 1000  # define its settings
       >>> problem = {'fitness_function': rosenbrock,  # cost function
       ...            'ndim_problem': ndim_problem,  # dimension
       ...            'lower_boundary': -10.0*np.ones((ndim_problem,)),  # search boundary
       ...            'upper_boundary': 10.0*np.ones((ndim_problem,))}

When the fitness function itself involves other *input arguments* except the sampling point `x`, there are
two simple ways to support this scenario:

* to create a `class <https://docs.python.org/3/reference/compound_stmts.html#class-definitions>`_ wrapper, e.g.:

    .. code-block:: python
       :linenos:

       >>> import numpy as np
       >>> def rosenbrock(x, arg):  # define the fitness (cost/objective) function
       ...     return arg*np.sum(np.power(x[1:] - np.power(x[:-1], 2), 2)) + np.sum(np.power(x[:-1] - 1, 2))
       >>> class Rosenbrock(object):  # build a class wrapper
       ...     def __init__(self, arg):  # arg is an extra input argument
       ...         self.arg = arg
       ...     def __call__(self, x):  # for fitness evaluation
       ...         return rosenbrock(x, self.arg)
       >>> rosen = Rosenbrock(100.0)
       >>> ndim_problem = 1000  # define its settings
       >>> problem = {'fitness_function': rosen,  # cost function
       ...            'ndim_problem': ndim_problem,  # dimension
       ...            'lower_boundary': -10.0*np.ones((ndim_problem,)),  # search boundary
       ...            'upper_boundary': 10.0*np.ones((ndim_problem,))}

* to utilize the easy-to-use unified interface provided for all optimizers in this library:

    .. code-block:: python
       :linenos:

       >>> import numpy as np
       >>> def rosenbrock(x, args):
       ...     return args*np.sum(np.power(x[1:] - np.power(x[:-1], 2), 2)) + np.sum(np.power(x[:-1] - 1, 2))
       >>> ndim_problem = 10
       >>> problem = {'fitness_function': rosenbrock,
       ...            'ndim_problem': ndim_problem,
       ...            'lower_boundary': -5*np.ones((ndim_problem,)),
       ...            'upper_boundary': 5*np.ones((ndim_problem,))}
       >>> from pypop7.optimizers.es.maes import MAES  # which can be replaced by any other optimizer in this library
       >>> options = {'fitness_threshold': 1e-10,  # terminate when the best-so-far fitness is lower than 1e-10
       ...            'max_function_evaluations': ndim_problem*10000,  # maximum of function evaluations
       ...            'seed_rng': 0,  # seed of random number generation (which must be set for repeatability)
       ...            'sigma': 3.0,  # initial global step-size of Gaussian search distribution
       ...            'verbose': 500}  # to print verbose information every 500 generations
       >>> maes = MAES(problem, options)  # initialize the optimizer
       >>> results = maes.optimize(args=100.0)  # args as input arguments of fitness function except sampling point
       >>> print(results['best_so_far_y'], results['n_function_evaluations'])
       3.98657911234714 100000  # this is a well-recognized *local* attractor rather than the global optimum

Optimizer Setting
-----------------

This library provides a *unified* API for hyper-parameter settings of all black-box optimizers. The following
algorithm options (all stored into a `dict`) are common for all optimizers:
  * `max_function_evaluations`: maximum of function evaluations (`int`, default: `np.Inf`),
  * `max_runtime`: maximal runtime to be allowed (`float`, default: `np.Inf`),
  * `seed_rng`: seed for random number generation needed to be *explicitly* set (`int`).

At least one of two options (`max_function_evaluations` and `max_runtime`) should be set, according to
the available computing resources or acceptable runtime.

For **repeatability**, `seed_rng` should be *explicitly* set for random number generation (`RNG
<https://numpy.org/doc/stable/reference/random/>`_).

Note that for any optimizer, its *specific* options/settings (see its API documentation for details) can be
naturally added into the `dict` data structure. Take the well-known `Cross-Entropy Method (CEM)
<https://link.springer.com/article/10.1007/s11009-006-9753-0>`_ as an illustrative example. The settings of
*mean* and *std* of its Gaussian sampling distribution usually have a significant impact on the convergence
rate (see its `API <https://pypop.readthedocs.io/en/latest/cem/scem.html>`_ for more details about its
hyper-parameters):

    .. code-block:: python
       :linenos:

       >>> import numpy as np
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.cem.scem import SCEM
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 10,
       ...            'lower_boundary': -5*np.ones((10,)),
       ...            'upper_boundary': 5*np.ones((10,))}
       >>> options = {'max_function_evaluations': 1000000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 4*np.ones((10,)),  # initial mean of Gaussian search distribution
       ...            'sigma': 3.0}  # initial std (aka global step-size) of Gaussian search distribution
       >>> scem = SCEM(problem, options)  # initialize the optimizer class
       >>> results = scem.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"SCEM: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       SCEM: 1000000, 10.328016143160333
