.. This is the official documentation of PyPop7 (a Pure-PYthon library of POPulation-based black-box OPtimization).

Welcome to PyPop7's Online Documentation for Black-Box Optimization!
====================================================================

.. image:: https://img.shields.io/badge/GitHub-PyPop7-red.svg
   :target: https://github.com/Evolutionary-Intelligence/pypop
.. image:: https://img.shields.io/badge/PyPI-pypop7-yellowgreen.svg
   :target: https://pypi.org/project/pypop7/
.. image:: https://img.shields.io/badge/license-GNU%20GPL--v3.0-green.svg
   :target: https://github.com/Evolutionary-Intelligence/pypop/blob/main/LICENSE
.. image:: https://img.shields.io/badge/OS-Linux%20%7C%20Windows%20%7C%20MacOS%20X-orange.svg
.. image:: https://readthedocs.org/projects/pypop/badge/?version=latest
.. image:: https://static.pepy.tech/badge/pypop7
   :target: https://www.pepy.tech/projects/pypop7
.. image:: https://img.shields.io/badge/Python-3-yellow.svg
.. image:: https://img.shields.io/badge/arxiv-2212.05652-red
   :target: https://arxiv.org/abs/2212.05652

**PyPop7** is a *Pure-PYthon* library of *POPulation-based OPtimization* for single-objective, real-parameter,
black-box problems. Its design goal is to provide a *unified* interface and *elegant* implementations for
**Black-Box Optimizers (BBO)**, *particularly population-based optimizers*, in order to facilitate research
repeatability, benchmarking of BBO, and also real-world applications.

Specifically, for alleviating the notorious **curse of dimensionality** of BBO, the primary focus of `PyPop7
<https://github.com/Evolutionary-Intelligence/pypop>`_ is to cover their **State-Of-The-Art (SOTA) implementations
for Large-Scale Optimization (LSO)**, though many of their *medium/small-scale* versions and variants are also included
here (some mainly for *theoretical* purposes, some mainly for *benchmarking* purposes, and some mainly for
*application* purposes on medium/low dimensions).

.. image:: images/logo.png
   :width: 321px
   :align: center

.. note::
   This `open-source <https://www.gnu.org/>`_ Python library for **continuous** BBO is under active maintenance (from 2023).
   In the future, we plan to add some NEW BBO and some SOTA versions of existing BBO, to make this library as fresh as possible.

**Quick Start**

In practice, three simple steps are enough to utilize the black-box optimization power of
`PyPop7 <https://pypi.org/project/pypop7/>`_:

1. Use `pip <https://pypi.org/project/pip/>`_ to automatically install `pypop7` via `PyPI <https://pypi.org/>`_:

    .. code-block:: bash

       $ pip install pypop7

See `this online documentation <https://pypop.readthedocs.io/en/latest/installation.html>`_ for details about *multiple*
installation ways.

2. Define your own objective/cost function (to be **minimized**) for the optimization problem at hand:

    .. code-block:: python
       :linenos:

       >>> import numpy as np  # for numerical computation, which is also the computing engine of pypop7
       >>> def rosenbrock(x):  # notorious test (benchmarking) function in the optimization community
       ...     return 100*np.sum(np.power(x[1:] - np.power(x[:-1], 2), 2)) + np.sum(np.power(x[:-1] - 1, 2))
       >>> ndim_problem = 1000  # problem dimension
       >>> problem = {'fitness_function': rosenbrock,  # cost function to be minimized
       ...            'ndim_problem': ndim_problem,  # problem dimension
       ...            'lower_boundary': -5.0*np.ones((ndim_problem,)),  # lower search boundary
       ...            'upper_boundary': 5.0*np.ones((ndim_problem,))}  # upper search boundary

See `this online documentation <https://pypop.readthedocs.io/en/latest/user-guide.html>`_ for details about the **problem
definition**. Note that any *maximization* problem can be transformed into the *minimization* problem via simply negating
it.

3. Run one or more black-box optimizers from `pypop7` on the above optimization problem:

    .. code-block:: python
       :linenos:

       >>> from pypop7.optimizers.es.lmmaes import LMMAES  # choose any optimizer you prefer in this library
       >>> options = {'fitness_threshold': 1e-10,  # terminate when the best-so-far fitness is lower than 1e-10
       ...            'max_runtime': 3600,  # terminate when the actual runtime exceeds 1 hour (i.e. 3600 seconds)
       ...            'seed_rng': 0,  # seed of random number generation (which must be set for repeatability)
       ...            'x': 4.0*np.ones((ndim_problem,)),  # initial mean of search/mutation distribution
       ...            'sigma': 3.0,  # initial global step-size of search distribution (to be fine-tuned)
       ...            'verbose': 500}
       >>> lmmaes = LMMAES(problem, options)  # initialize the optimizer (a unified interface for all optimizers)
       >>> results = lmmaes.optimize()  # run its (time-consuming) search process
       >>> # print the best-so-far fitness and used function evaluations returned by the used black-box optimizer
       >>> print(results['best_so_far_y'], results['n_function_evaluations'])
       9.948e-11 2973386

See `this online documentation <https://pypop.readthedocs.io/en/latest/user-guide.html#optimizer-setting>`_ for
details about the **optimizer setting**. Please refer to the following contents for all the BBO available in this
open-source library.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   design-philosophy
   user-guide
   tutorials/tutorials
   es/es
   nes/nes
   eda/eda
   cem/cem
   de/de
   pso/pso
   cc/cc
   sa/sa
   ga/ga
   ep/ep
   ds/ds
   rs/rs
   bo/bo
   bbo
   development-guide
   software-summary
   applications
   sponsor
   changing-log
