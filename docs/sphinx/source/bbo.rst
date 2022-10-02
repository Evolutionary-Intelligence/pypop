Black-Box Optimization (BBO)
============================

.. note:: *"Certainly, and especially because of the broad availability of difficult and important applications, this
   promises to be an exciting, interesting, and challenging area for many years to come."* ---[Conn et al., 2009:
   INTRODUCTION TO DERIVATIVE-FREE OPTIMIZATION]

The **black-box** nature of many real-world optimization problems comes from one or more of the following factors,
as shown in e.g. the classical book [INTRODUCTION TO DERIVATIVE-FREE OPTIMIZATION]:

* increasing complexity in mathematical modeling,
* higher sophistication of scientific computing,
* an abundance of legacy or proprietary codes,
* noisy function evaluations.

No Free Lunch Theorems (NFL)
----------------------------

As mathematically proved in `[Wolpert&Macready, 1997, TEVC] <https://ieeexplore.ieee.org/document/585893>`_, **"for any
algorithm, any elevated performance over one class of problems is offset by performance over another class."**

Curse of Dimensionality for Large-Scale BBO (LSBBO)
---------------------------------------------------

POPulation-based OPtimization/Search (POP)
------------------------------------------

.. note:: *"The essence of an evolutionary approach to solve a problem is to equate possible solutions to individuals
   in a population, and to introduce a notion of fitness on the basis of solution quality."* ---`[Eiben&Smith, 2015,
   Nature] <https://www.nature.com/articles/nature14544>`_

General-Purpose Optimization Algorithms
---------------------------------------

.. note:: *"Given the abundance of black-box optimization algorithms and of optimization problems, how can best match
   algorithms to problems."* ---`[Wolpert&Macready, 1997, TEVC] <https://ieeexplore.ieee.org/document/585893>`_

Limitations of BBO
------------------

.. note:: *"If you can obtain clean derivatives (even if it requires considerable effort) and the functions defining
   your problem are smooth and free of noise you should not use derivative-free methods.."* ---[Conn et al., 2009:
   INTRODUCTION TO DERIVATIVE-FREE OPTIMIZATION]
