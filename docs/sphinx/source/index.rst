.. This is the official documents of PyPop7 (Pure-PYthon library of POPulation-based black-box OPtimization).

Welcome to PyPop7's Documentation!
==================================

.. image:: https://img.shields.io/badge/GitHub-PyPop7-red.svg
.. image:: https://img.shields.io/badge/PyPI-pypop7-yellowgreen.svg
.. image:: https://img.shields.io/badge/license-GNU%20GPL--v3.0-green.svg
.. image:: https://readthedocs.org/projects/pypop/badge/?version=latest
.. image:: https://pepy.tech/badge/pypop7

**PyPop7** is a *Pure-PYthon* library of *POPulation-based OPtimization* for single-objective, real-parameter, black-box problems. Its main goal is to provide a *unified* interface and *elegant* implementations for **Black-Box Optimizers (BBO)**, *particularly population-based optimizers*, in order to facilitate research repeatability and also real-world applications.

More specifically, for alleviating the notorious **curse of dimensionality** of BBO (almost based on *iterative sampling*), the primary focus of PyPop7 is to cover their **State-Of-The-Art (SOTA) implementations for Large-Scale Optimization (LSO)**, though many of their other versions and variants are also included here (for benchmarking/mixing purpose, and sometimes even for practical purpose).

.. image:: images/logo.png
   :width: 321px
   :align: center

.. note::
   Now this library is still under active development.

**Quick Start**

Three simple steps are enough to utilize the optimization power of `PyPop7 <https://pypi.org/project/pypop7/>`_:

First, use `pip <https://pypi.org/project/pip/>`_ to automatically install `pypop7`:

    .. code-block:: bash

       pip install pypop7

Then define your own objective function for the optimization problem at hand, and
last run one or more black-box optimizers from `pypop7` on the given optimization problem:

    .. code-block:: python
       :linenos:

       >>> import numpy as np  # for numerical computation, which is also the computing engine of pypop7
       >>> def rosenbrock(x):  # the notorious test function in the optimization community
       ...     return 100 * np.sum(np.power(x[1:] - np.power(x[:-1], 2), 2)) + np.sum(np.power(x[:-1] - 1, 2))
       >>> ndim_problem = 1000  # define the fitness (cost) function and also its settings
       >>> problem = {'fitness_function': rosenbrock,  # cost function
       ...            'ndim_problem': ndim_problem,  # dimension
       ...            'lower_boundary': -5 * np.ones((ndim_problem,)),  # search boundary
       ...            'upper_boundary': 5 * np.ones((ndim_problem,))}
       >>> from pypop7.optimizers.es.lmmaes import LMMAES  # to choose any optimizer you prefer in this library
       >>> options = {'fitness_threshold': 1e-10,  # terminate when the best-so-far fitness is lower than 1e-10
       ...            'max_runtime': 3600,  # terminate when the actual runtime exceeds 1 hour (i.e. 3600 seconds)
       ...            'seed_rng': 0,  # seed of random number generation (which must be set for repeatability)
       ...            'x': 4 * np.ones((ndim_problem,)),  # initial mean of search (mutation/sampling) distribution
       ...            'sigma': 0.3,  # initial global step-size of search distribution
       ...            'verbose_frequency': 500}
       >>> lmmaes = LMMAES(problem, options)  # initialize the optimizer
       >>> results = lmmaes.optimize()  # run its (time-consuming) search process
       >>> # print the best-so-far fitness and used function evaluations returned by the black-box optimizer
       >>> print(results['best_so_far_y'], results['n_function_evaluations'])
       9.8774e-11 3928055



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   design-philosophy
   es/es
   es/res
   es/ssaes
   es/dsaes
   es/csaes
   es/saes
   es/maes
   es/fmaes
   es/lmcmaes
   es/lmcma
   es/lmmaes
   es/r1es
   eda/eda
   eda/umda
   eda/emna
   de/de
   de/cde
   de/tde
   de/jade
   de/code
   ep/ep
   ep/cep
   ep/fep
   ds/ds
   ds/cs
   ds/hj
   ds/nm
   rs/rs
   rs/prs
   rs/rhc
   rs/arhc
   rs/srs
   sponsor
