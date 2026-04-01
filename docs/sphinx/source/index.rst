PyPop7's Documentations for Continuous Black-Box Optimization (BBO)
===================================================================

**[NEWS]** Recently PyPop7 has been used and/or cited in one
**Nature** paper (`[Veenstra et al., Nature, 2025]
<https://www.nature.com/articles/s41586-025-08646-3>`_), etc.

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
.. image:: https://visitor-badge.laobi.icu/badge?page_id=Evolutionary-Intelligence.pypop
   :target: https://visitor-badge.laobi.icu/badge?page_id=Evolutionary-Intelligence.pypop
.. image:: https://img.shields.io/badge/arxiv-2212.05652-red
   :target: https://arxiv.org/abs/2212.05652
.. image:: https://img.shields.io/badge/JMLR-2024-red
   :target: https://jmlr.org/
.. image:: https://readthedocs.org/projects/pypop/badge/?version=latest
   :target: https://readthedocs.org/projects/pypop/

`"Responsible for adaptation, optimization, and innovation in the living world, evolution
executes a simple algorithm of diversifcation and natural selection, an algorithm that
works at all levels of complexity from single protein molecules to whole ecosystems."---
From Nobel Lecture of Frances H. Arnold in California Institute of Technology
<https://www.nobelprize.org/uploads/2018/10/arnold-lecture.pdf>`_

**PyPop7** is a *Pure-PYthon* library of *POPulation-based OPtimization* for single-objective,
real-parameter, black-box problems. Its design goal is to provide a *unified* interface and a
set of *elegant* implementations for **Black-Box Optimizers (BBO)**, particularly
**population-based optimizers** (including *evolutionary algorithms, swarm-based random
methods, and pattern search*), in order to facilitate research repeatability, benchmarking of
BBO, and especially real-world applications.

Specifically, for alleviating the well-known ("notorious") **curse of dimensionality** of BBO,
the primary focus of `PyPop7 <https://github.com/Evolutionary-Intelligence/pypop>`_ is to
cover their **State-Of-The-Art (SOTA) implementations for Large-Scale Optimization (LSO)** as
much as possible, though many of their *medium- or small-scale* versions and variants are also
included here (some mainly for *theoretical* purposes, some mainly for *educational* purposes,
some mainly for *benchmarking* purposes, and some mainly for *application* purposes on medium
or low dimensions).

.. note::
   This `open-source <https://www.gnu.org/>`_ Python library for **continuous** BBO is still under active maintenance.
   In the future, we plan to add some NEW BBO and some SOTA versions of existing BBO families, in order to make this
   library as fresh as possible. Any suggestions, extensions, improvements, usages, and tests (even *criticisms*) to
   this `open-source <https://opensource.org/>`_ Python library are highly welcomed!

   Now this arXiv paper has been submitted to `JMLR
   <https://jmlr.org/>`_, **accepted** in Fri, 11
   Oct 2024 after 3 reviews from Tue, 28 Mar 2023
   to Wed, 01 Nov 2023 to Fri, 05 Jul 2024.)

**Quick Start**

Three steps are often enough to utilize the potential of `PyPop7 <https://pypi.org/project/pypop7/>`_ for BBO in many
(though not all) cases:

1. Use `pip <https://pypi.org/project/pip/>`_ to automatically install `pypop7` via `PyPI <https://pypi.org/>`_:

    .. code-block:: bash

       $ pip install pypop7

Please refer to `this online documentation <https://pypop.readthedocs.io/en/latest/installation.html>`_ for details
about *multiple* installation ways.

2. Define your own *objective* (aka *cost* or *fitness*)
   function to be **minimized** for the `complex
   <https://doi.org/10.1201/9780367802486>`_
   optimization problem at hand:

    .. code-block:: python
       :linenos:

       >>> import numpy as np  # for numerical computation (PyPop7's computing engine)
       >>> def rosenbrock(x):  # one notorious function in the optimization community
       ...     return 100.0*np.sum(np.square(x[1:] - np.square(x[:-1]))) + np.sum(np.square(x[:-1] - 1.0))
       >>> ndim_problem = 1000  # problem dimension
       >>> problem = {'fitness_function': rosenbrock,  # fitness function to be minimized
       ...            'ndim_problem': ndim_problem,  # problem dimension
       ...            'lower_boundary': -5.0*np.ones((ndim_problem,)),  # lower search boundary
       ...            'upper_boundary': 5.0*np.ones((ndim_problem,))}  # upper search boundary

Please refer to `this online documentation
<https://pypop.readthedocs.io/en/latest/user-guide.html#id1>`_
for details about **problem definition**. Note that any
*maximization* problem can be transformed into the
*minimization* problem simply via negating it.

3. Run one black-box optimizer or more from
   `PyPop7` on the above problem:

    .. code-block:: python
       :linenos:

       >>> from pypop7.optimizers.es.lmmaes import LMMAES  # choose any optimizer which you prefer
       >>> options = {'fitness_threshold': 1e-10,  # terminate when the best-so-far fitness < 1e-10
       ...            'max_runtime': 3600,  # terminate when the runtime exceeds 1 hour
       ...            'seed_rng': 0,  # seed of random number generation (for repeatability)
       ...            'x': 4.0*np.ones((ndim_problem,)),  # initial mean of search distribution
       ...            'sigma': 3.0,  # initial global step-size (to be fine-tuned for optimality)
       ...            'verbose': 500}
       >>> lmmaes = LMMAES(problem, options)  # initialize the optimizer (a unified interface)
       >>> results = lmmaes.optimize()  # run its (time-consuming) optimization (evolution) process
       >>> # print best-so-far fitness and used function evaluations returned by the optimizer
       >>> print(results['best_so_far_y'], results['n_function_evaluations'])
       9.948e-11 2973386 (# different NumPy versions may result in different results #)

Please refer to `this online documentation
<https://pypop.readthedocs.io/en/latest/user-guide.html#id2>`_
for details about **optimizer settings**.

.. note::
   If this open-source Python library is used in your project or paper, please
   cite the following **JMLR** paper (*BibTeX*):

   @article{2024-JMLR-Duan,
   author={Duan, Qiqi and Zhou, Guochen and Shao, Chang and Others},
   title={{PyPop7}: A {pure-Python} library for population-based black-box optimization},
   journal={Journal of Machine Learning Research},
   volume={25},
   number={296},
   pages={1--28},
   year={2024}
   }



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
   design-philosophy
   activities
   how-to-cite-pypop7
   stars
   history
   reference
