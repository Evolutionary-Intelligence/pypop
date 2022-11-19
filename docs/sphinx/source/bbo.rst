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

Some common problem characteristics of BBO are presented below:

* non-differentiability,
* non-linearity,
* multi-modality,
* ill-condition,
* noisiness,
* multiple objectives (though not covered in this library).

No Free Lunch Theorems (NFL)
----------------------------

As mathematically proved in `[Wolpert&Macready, 1997, TEVC] <https://ieeexplore.ieee.org/document/585893>`_, **"for any
algorithm, any elevated performance over one class of problems is offset by performance over another class."**

Curse of Dimensionality for Large-Scale BBO (LSBBO)
---------------------------------------------------

General-Purpose Optimization Algorithms
---------------------------------------

.. note:: *"Given the abundance of black-box optimization algorithms and of optimization problems, how can best match
   algorithms to problems."* ---`[Wolpert&Macready, 1997, TEVC] <https://ieeexplore.ieee.org/document/585893>`_

Some of the following criteria/principles may be highly expected to satisfy for general-purpose optimization algorithms:

* beauty (arguably from theoretical depth),
* effectiveness,
* efficiency,
* elegance,
* scalability,
* simplicity,
* versatility,
* width of applications.

POPulation-based OPtimization/Search (POP)
------------------------------------------

.. note:: *"The essence of an evolutionary approach to solve a problem is to equate possible solutions to individuals
   in a population, and to introduce a notion of fitness on the basis of solution quality."* ---`[Eiben&Smith, 2015,
   Nature] <https://www.nature.com/articles/nature14544>`_

Population-based (particularly evolutionary) optimizers (POP) usually have the following advantages, when compared to individual-based counterparts:

* few assumptions (even assumptions-free),
* flexible framework (easy integration with problem-specific knowledge),
* robust performance (e.g. w.r.t. noisy perturbation or hyper-parameters),
* diverse solutions (e.g. for multi-modal/multi-objective/dynamic optimization),
* novelty (e.g. beyond intuitions for design problems).

Limitations of BBO
------------------

.. note:: *"If you can obtain clean derivatives (even if it requires considerable effort) and the functions defining
   your problem are smooth and free of noise you should not use derivative-free methods.."* ---[Conn et al., 2009:
   INTRODUCTION TO DERIVATIVE-FREE OPTIMIZATION]

Importantly, **not all** optimization problems fit in black-box optimizers. In fact, its arbitrary abuse in science and
engineering can lead to wide criticism. Although not always, black-box optimizers often be seen as *"the last choice of
search methods"*.
