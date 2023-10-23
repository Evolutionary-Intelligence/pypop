Development Guide
=================

.. note::
   This `Development Guide` page is still actively updated. We wish to make **adding new optimizers**
   as easy as possible.

Before reading this page, it is required to first read `User Guide
<https://pypop.readthedocs.io/en/latest/user-guide.html>`_ for some basic information about this
open-source Python library `PyPop7`. Note that since this topic is mainly for advanced developers,
the end-users can skip this page freely.

Docstring Conventions
---------------------

For **docstring conventions**, first `PEP 257 <https://peps.python.org/pep-0257/>`_ is used in this library.
Since this library is built on the `NumPy <https://www.nature.com/articles/s41586-020-2649-2>`_ ecosystem,
we further use the docstring conventions from
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

A Unified API
-------------

For `PyPop7`, we use the popular Object-Oriented Programming (OOP) paradigm to structure all optimizers, which
can provide consistency, flexibility, and simplicity. We did not adopt another popular
Procedure-Oriented Programming paradigm. However, in the future versions, we may provide such an interface
only at the end-user level (rather than the developer level).

For all optimizers, the abstract class called `Optimizer
<https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/core/optimizer.py>`_
needs to be inherited, in order to provide a unified API.

* All members shared by all optimizers (e.g., `fitness_function`, `ndim_problem`, etc.) should be
  defined in the `__init__
  <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/core/optimizer.py#L41>`_
  method of this class.

* All methods public to end-users should be defined in this class except special cases.

* All settings related to fair benchmarking comparisons (e.g., `max_function_evaluations`,
  `max_runtime`, and `fitness_threshold`) should be defined in the `__init__
  <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/core/optimizer.py#L41>`_
  method of this class.

Initialization of Optimizer Options
-----------------------------------

For initialization of optimizer options, the following function `__init__
<https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/core/optimizer.py#L41>`_
of `Optimizer` should be inherited:

    .. code-block:: bash

       def __init__(self, problem, options):
           # here all members will be inherited by any subclass of `Optimizer`

All *exclusive* members of each subclass will be defined after inheriting the above function of `Optimizer`.

Initialization of Population
----------------------------

We separate the initialization of *optimizer options* with that of *population* (a set of individuals),
in order to obtain flexibility. To achieve this, the following function `initialize
<https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/core/optimizer.py#L147>`_ should
be modified:

    .. code-block:: bash

       def initialize(self):  # for population initialization
           raise NotImplementedError  # need to be implemented in any subclass of `Optimizer`

Its another goal is to minimize the number of class members, to make it easy to set for end-users,
but at a slight cost of more variables control for developers.

Computation of Each Generation
------------------------------

Update each one generation (iteration) via modifying the following function `iterate
<https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/core/optimizer.py#L150>`_:

    .. code-block:: bash

       def iterate(self):  # for one generation (iteration)
           raise NotImplementedError  # need to be implemented in any subclass of `Optimizer`

Control of Entire Optimization Process
--------------------------------------

Control the entire search process via modifying the following function `optimize
<https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/core/optimizer.py#L153>`_:

    .. code-block:: bash

       def optimize(self, fitness_function=None):  # entire optimization process
           return None  # `None` should be replaced in any subclass of `Optimizer`

Typically, common auxiliary tasks (e.g., printing verbose information, restarting) are conducted inside
this function.

Using Pure Random Search as an Illustrative Example
---------------------------------------------------

In the following Python code, we use Pure Random Search (PRS), perhaps the simplest black-box optimizer, as
an illustrative example.

   .. code-block:: bash

      import numpy as np
      
      from pypop7.optimizers.core.optimizer import Optimizer  # base class of all black-box optimizers
 
      
      class PRS(Optimizer):
          """Pure Random Search (PRS).

          .. note:: `PRS` is one of the *simplest* and *earliest* black-box optimizers, dating back to at least
             `1950s <https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244>`_.
             Here we include it mainly for *benchmarking* purpose. As pointed out in `Probabilistic Machine Learning
             <https://probml.github.io/pml-book/book2.html>`_, *this should always be tried as a baseline*.
      
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
                    and with the following particular setting (`key`):
                      * 'x' - initial (starting) point (`array_like`).
      
          Attributes
          ----------
          x     : `array_like`
                  initial (starting) point.
      
          Examples
          --------
          Use the `PRS` optimizer to minimize the well-known test function
          `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:
      
          .. code-block:: python
             :linenos:
      
             >>> import numpy
             >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
             >>> from pypop7.optimizers.rs.prs import PRS
             >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
             ...            'ndim_problem': 2,
             ...            'lower_boundary': -5.0*numpy.ones((2,)),
             ...            'upper_boundary': 5.0*numpy.ones((2,))}
             >>> options = {'max_function_evaluations': 5000,  # set optimizer options
             ...            'seed_rng': 2022}
             >>> prs = PRS(problem, options)  # initialize the optimizer class
             >>> results = prs.optimize()  # run the optimization process
             >>> print(results)
      
          For its correctness checking of coding, refer to `this code-based repeatability report
          <https://tinyurl.com/mrx2kffy>`_ for more details.
      
          References
          ----------
          Bergstra, J. and Bengio, Y., 2012.
          Random search for hyper-parameter optimization.
          Journal of Machine Learning Research, 13(2).
          https://www.jmlr.org/papers/v13/bergstra12a.html
      
          Schmidhuber, J., Hochreiter, S. and Bengio, Y., 2001.
          Evaluating benchmark problems by random guessing.
          A Field Guide to Dynamical Recurrent Networks, pp.231-235.
          https://ml.jku.at/publications/older/ch9.pdf
      
          Brooks, S.H., 1958.
          A discussion of random methods for seeking maxima.
          Operations Research, 6(2), pp.244-251.
          https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244
          """
          def __init__(self, problem, options):
              """Initialize the class with two inputs (problem arguments and optimizer options)."""
              Optimizer.__init__(self, problem, options)
              self.x = options.get('x')  # initial (starting) point
              self.verbose = options.get('verbose', 1000)
              self._n_generations = 0  # number of generations
      
          def _sample(self, rng):
              x = rng.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
              return x
      
          def initialize(self):
              """Only for the initialization stage."""
              if self.x is None:
                  x = self._sample(self.rng_initialization)
              else:
                  x = np.copy(self.x)
              assert len(x) == self.ndim_problem
              return x

          def iterate(self):
              """Only for the iteration stage."""
              return self._sample(self.rng_optimization)

          def _print_verbose_info(self, fitness, y):
              """Save fitness and control console verbose information."""
              if self.saving_fitness:
                  if not np.isscalar(y):
                      fitness.extend(y)
                  else:
                      fitness.append(y)
              if self.verbose and ((not self._n_generations % self.verbose) or (self.termination_signal > 0)):
                  info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
                  print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))
       
          def _collect(self, fitness, y=None):
              """Collect necessary output information."""
              if y is not None:
                  self._print_verbose_info(fitness, y)
              results = Optimizer._collect(self, fitness)
              results['_n_generations'] = self._n_generations
              return results

          def optimize(self, fitness_function=None, args=None):  # for all iterations (generations)
              """For the entire optimization/evolution stage: initialization + iteration."""
              fitness = Optimizer.optimize(self, fitness_function)
              x = self.initialize()  # population initialization
              y = self._evaluate_fitness(x, args)  # to evaluate fitness of starting point
              while not self._check_terminations():
                  self._print_verbose_info(fitness, y)  # to save fitness and control console verbose information
                  x = self.iterate()
                  y = self._evaluate_fitness(x, args)  # to evaluate each new point
                  self._n_generations += 1
              results = self._collect(fitness, y)  # to collect all necessary output information 
              return results

Note that from Oct. 22, 2023, we have decided to adopt the *active* development/maintenance mode, that is, **once
new optimizers are added or serious bugs are fixed, we will release a new version right now**.

Repeatability Code/Reports
--------------------------

=========== ================================================================================================================================== ==============================================================================================================
 Optimizer   Repeatability Code                                                                                                                Generated Figure(s)/Data                                                                          
=========== ================================================================================================================================== ==============================================================================================================
 MMES          `_repeat_mmes.py <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_mmes.py>`_          `figures <https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/mmes>`_  

 FCMAES     `_repear_fcmaes.py <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_fcmaes.py>`_         `figures <https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/fcmaes>`_

 LMMAES     `_repeat_lmmaes.py <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_lmmaes.py>`_         `figures <https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/lmmaes>`_

 LMCMA      `_repeat_lmcma.py <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_lmcma.py>`_           `figures <https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/lmcma>`_

 LMCMAES    `_repeat_lmcmaes.py <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_lmcmaes.py>`_       `data <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_lmcmaes.py>`_

 RMES       `_repeat_rmes.py <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_rmes.py>`_             `figures <https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/rmes>`_

 R1ES       `_repeat_r1es.py <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_r1es.py>`_             `figures <https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/r1es>`_

 VKDCMA     `_repeat_vkdcma.py <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_vkdcma.py>`_         `data <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_vkdcma.py>`_

 VDCMA      `_repeat_vdcma.py <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_vdcma.py>`_           `data <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_vdcma.py>`_

 CCMAES2016 `_repeat_ccmaes2016.py <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_ccmaes2016.py>`_ `figures <https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/ccmaes2016>`_
=========== ================================================================================================================================== ==============================================================================================================
