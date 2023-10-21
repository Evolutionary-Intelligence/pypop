Development Guide
=================

.. note::
   This `Development Guide` page is still actively updated.

Before reading this page, it is required to first read `User Guide
<https://pypop.readthedocs.io/en/latest/user-guide.html>`_ for some basic information. Note that
since this topic is mainly for advanced developers, the end-users can skip this page freely.

Docstring Conventions
---------------------

For **docstring conventions**, `PEP 257 <https://peps.python.org/pep-0257/>`_ is used in this library.
Since this library is built on the `NumPy <https://www.nature.com/articles/s41586-020-2649-2>`_ ecosystem,
we further use the docstring conventions from
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

A Unified API
-------------

For `PyPop7`, we use the popular Object-Oriented Programming (OOP) to structure all optimizers, which
can provide consistency, flexibility, and simplicity. We did not adopt another popular
Procedure-Oriented Programming. However, in the future versions, we may provide such an interface
only at the end-user level (rather than the developer level).

For all optimizers, the abstract class called `Optimizer
<https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/core/optimizer.py>`_
needs to be inherited implicitly or explicitly, in order to provide a unified API.

* All members shared by all optimizers (e.g., `fitness_function`, `ndim_problem`, etc.) should be
  defined in this class.

* All functions public to end-users should be defined in this class except special cases.

* All settings related to fair benchmarking comparisons (e.g., `max_function_evaluations`,
  `max_runtime`, and `fitness_threshold`) should be defined in this class.

Initialization of Optimizer Options
-----------------------------------

For initialization of optimizer options, the following function `__init__
<https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/core/optimizer.py#L41>`_
of `Optimizer` should be inherited:

    .. code-block:: bash

       def __init__(self, problem, options):
           # here all members will be inherited by any subclass of `Optimizer`

The *exclusive* members of the subclass will be defined after inheriting the above function of `Optimizer`.

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

   .. code-block:: bash

         import numpy as np
         
         from pypop7.optimizers.core.optimizer import Optimizer
         
         
         class PRS(Optimizer):  # inherited the abstract class called `Optimizer`
             """Pure Random Search (PRS).
             """
             def __init__(self, problem, options):
                 Optimizer.__init__(self, problem, options)
                 self.x = options.get('x')  # initial (starting) point
                 self._n_generations = 0  # number of generations
         
             def _sample(self, rng):
                 x = rng.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
                 return x
         
             # Initialization of Population
             def initialize(self):
                 if self.x is None:
                     x = self._sample(self.rng_initialization)
                 else:
                     x = np.copy(self.x)
                 assert len(x) == self.ndim_problem
                 return x
         
             # Computation of Each Generation
             def iterate(self): # individual-based sampling
                 return self._sample(self.rng_optimization)
         
             # Saving of Finess and Control of Output Verbose Information
             def _print_verbose_info(self, fitness, y):
                 if self.saving_fitness:
                     if not np.isscalar(y):
                         fitness.extend(y)
                     else:
                         fitness.append(y)
                 if self.verbose and ((not self._n_generations % self.verbose) or (self.termination_signal > 0)):
                     info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
                     print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))
         
             # Collection of Necessary Information 
             def _collect(self, fitness, y=None):
                 if y is not None:
                     self._print_verbose_info(fitness, y)
                 results = Optimizer._collect(self, fitness)
                 results['_n_generations'] = self._n_generations
                 return results
         
             # Control of Entire Optimization Process
             def optimize(self, fitness_function=None, args=None):  # for all iterations (generations)
                 fitness = Optimizer.optimize(self, fitness_function)
                 x = self.initialize()  # Initialization of Population
                 y = self._evaluate_fitness(x, args)  # fitness of starting point
                 while not self._check_terminations():
                     self._print_verbose_info(fitness, y)  # Saving of Finess and  Control of Output Verbose Information
                     x = self.iterate()  # Computation of Each Generation
                     y = self._evaluate_fitness(x, args)  # to evaluate the new point
                     self._n_generations += 1
                 results = self._collect(fitness, y)  # Collection of Necessary Information 
                 return results

   
