Development Guide
=================

.. note::
   This `Development Guide` page is still actively updated.

Before reading this page, it is required to first read `User Guide
<https://pypop.readthedocs.io/en/latest/user-guide.html>`_ for some basic information. Note that
this topic is mainly for advanced developers, the end-users can skip this page freely.

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

Initialization of Optimizer Options
-----------------------------------

For initialization of optimizers, the following function of `Optimizer` should be inherited:

    .. code-block:: bash

       def __init__(self, problem, options):
           # here all members will be inherited by any its subclass

Initialization of Population
----------------------------

We separate the initialization of optimizer options with that of population (a set of individuals),
in order to obtain better flexibility:

    .. code-block:: bash

       def initialize(self):  # for population initialization
           raise NotImplementedError

Control of One Generation (Iteration)
-------------------------------------

Define each generation in the following function:

    .. code-block:: bash

       def iterate(self):  # for one generation
           raise NotImplementedError  # need to be implemented in any its subclass

Control of Entire Optimization Process
--------------------------------------

Control the entire search process via the following function:

    .. code-block:: bash

       def optimize(self, fitness_function=None):
           return None  # `None` should be replaced in any its subclass
