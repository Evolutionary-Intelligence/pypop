Documentations of PyPop7 for Black-Box Optimization (BBO)
=========================================================

.. image:: https://img.shields.io/badge/GitHub-PyPop7-red.svg
   :target: https://github.com/Evolutionary-Intelligence/pypop
.. image:: https://img.shields.io/badge/PyPI-pypop7-yellowgreen.svg
   :target: https://pypi.org/project/pypop7/
.. image:: https://img.shields.io/badge/license-GNU%20GPL--v3.0-green.svg
   :target: https://github.com/Evolutionary-Intelligence/pypop/blob/main/LICENSE
.. image:: https://img.shields.io/badge/OS-Linux%20%7C%20Windows%20%7C%20MacOS%20X-orange.svg
   :target: https://www.usenix.org/conferences/byname/179
.. image:: https://static.pepy.tech/badge/pypop7
   :target: https://www.pepy.tech/projects/pypop7
.. image:: https://static.pepy.tech/badge/pypop7/month
   :target: https://www.pepy.tech/projects/pypop7
.. image:: https://visitor-badge.laobi.icu/badge?page_id=Evolutionary-Intelligence.pypop
   :target: https://visitor-badge.laobi.icu/badge?page_id=Evolutionary-Intelligence.pypop
.. image:: https://img.shields.io/badge/arxiv-2212.05652-red
   :target: https://arxiv.org/abs/2212.05652
.. image:: https://readthedocs.org/projects/pypop/badge/?version=latest
   :target: https://readthedocs.org/projects/pypop/

**PyPop7** is a *Pure-PYthon* library of *POPulation-based OPtimization* for single-objective,
real-parameter, black-box problems. Its design goal is to provide a *unified* interface and a
set of *elegant* implementations for **Black-Box Optimizers (BBO)**, *particularly
population-based optimizers*, in order to facilitate research repeatability, benchmarking of
BBO, and especially real-world applications.

Specifically, for alleviating the well-known ("notorious") **curse of dimensionality** of BBO, the primary focus
of `PyPop7 <https://github.com/Evolutionary-Intelligence/pypop>`_ is to cover their **State-Of-The-Art (SOTA)
implementations for Large-Scale Optimization (LSO)** as much as possible, though many of their
*medium/small-scale* versions and variants are also included here (some mainly for *theoretical* purposes, some
mainly for *educational* purposes, some mainly for *benchmarking* purposes, and some mainly for *application*
purposes on medium/low dimensions).

.. image:: images/logo.png
   :width: 321px
   :align: center

.. note::
   This `open-source <https://www.gnu.org/>`_ Python library for **continuous** BBO is still under active maintenance.
   In the future, we plan to add some NEW BBO and some SOTA versions of existing BBO families, in order to make this
   library as fresh as possible. Any suggestions, extensions, improvements, usages, and tests (even *criticisms*) to
   this `open-source <https://opensource.org/>`_ Python library are highly welcomed!

   If this open-source pure-Python library **PyPop7** is used in your paper or project, it is highly welcomed but NOT
   mandatory to cite the following arXiv preprint paper: **Duan, Q., Zhou, G., Shao, C., Wang, Z., Feng, M., Huang,
   Y., Tan, Y., Yang, Y., Zhao, Q. and Shi, Y., 2024. PyPop7: A pure-Python library for population-based black-box
   optimization. arXiv preprint arXiv:2212.05652.** (Now this arXiv paper has been submitted to `JMLR
   <https://jmlr.org/>`_, under review).

**Quick Start**

Three steps are often enough to utilize the potential of `PyPop7 <https://pypi.org/project/pypop7/>`_ for BBO in many
(though not all) cases:

1. Use `pip <https://pypi.org/project/pip/>`_ to automatically install `pypop7` via `PyPI <https://pypi.org/>`_:

    .. code-block:: bash

       $ pip install pypop7

Please refer to `this online documentation <https://pypop.readthedocs.io/en/latest/installation.html>`_ for details
about *multiple* installation ways.

2. Define/code your own objective (aka cost or fitness) function (to be **minimized**) for the `complex
   <https://doi.org/10.1201/9780367802486>`_ optimization problem at hand:

    .. code-block:: python
       :linenos:

       >>> import numpy as np  # for numerical computation, which is also the computing engine used by PyPop7
       >>> def rosenbrock(x):  # one notorious test function in the optimization community
       ...     return 100.0*np.sum(np.square(x[1:] - np.square(x[:-1]))) + np.sum(np.square(x[:-1] - 1.0))
       >>> ndim_problem = 1000  # problem dimension
       >>> problem = {'fitness_function': rosenbrock,  # fitness function to be minimized
       ...            'ndim_problem': ndim_problem,  # problem dimension
       ...            'lower_boundary': -5.0*np.ones((ndim_problem,)),  # lower search boundary
       ...            'upper_boundary': 5.0*np.ones((ndim_problem,))}  # upper search boundary

See `this online documentation <https://pypop.readthedocs.io/en/latest/user-guide.html>`_ for details about the
**problem definition**. Note that any *maximization* problem can be easily transformed into the *minimization*
problem via simply negating.

Please refer to `this online documentation <https://pypop.readthedocs.io/en/latest/benchmarks.html>`_ for a
large set of benchmarking functions from different application fields, which have been provided by `PyPop7`.

3. Run one or more black-box optimizers (BBO) from `PyPop7` on the above optimization problem:

    .. code-block:: python
       :linenos:

       >>> from pypop7.optimizers.es.lmmaes import LMMAES  # or to choose any black-box optimizer you prefer in PyPop7
       >>> options = {'fitness_threshold': 1e-10,  # terminate when the best-so-far fitness is lower than 1e-10
       ...            'max_runtime': 3600,  # terminate when the actual runtime exceeds 1 hour (i.e., 3600 seconds)
       ...            'seed_rng': 0,  # seed of random number generation (which should be set for repeatability)
       ...            'x': 4.0*np.ones((ndim_problem,)),  # initial mean of search/mutation/sampling distribution
       ...            'sigma': 3.0,  # initial global step-size of search distribution (to be fine-tuned for optimality)
       ...            'verbose': 500}
       >>> lmmaes = LMMAES(problem, options)  # initialize the black-box optimizer (a unified interface for all optimizers)
       >>> results = lmmaes.optimize()  # run its (time-consuming) optimization/evolution/search process
       >>> # print best-so-far fitness and used function evaluations returned by the used black-box optimizer
       >>> print(results['best_so_far_y'], results['n_function_evaluations'])
       9.948e-11 2973386

Please refer to `this online documentation <https://pypop.readthedocs.io/en/latest/user-guide.html#optimizer-setting>`_
for details about the **optimizer setting**. The following **API** contents are given mainly for all currently available
BBO in this seemingly `increasingly popular <https://pypop.readthedocs.io/en/latest/applications.html>`_ open-source
Python library.



.. toctree::
   :maxdepth: 2
   :caption: Contents of PyPop7:

   installation
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
   benchmarks
   util-functions-for-BBO
   development-guide
   applications
   sponsor
   design-philosophy
   changing-log
   software-summary



.. image:: https://visitor-badge.laobi.icu/badge?page_id=Evolutionary-Intelligence.pypop
   :target: https://visitor-badge.laobi.icu/badge?page_id=Evolutionary-Intelligence.pypop
