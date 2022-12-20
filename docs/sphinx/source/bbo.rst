Black-Box Optimization (BBO)
============================

.. note:: `"Certainly, and especially because of the broad availability of difficult and important applications, this
   promises to be an exciting, interesting, and challenging area for many years to come."---[Conn et al., 2009,
   Introduction to Derivative-Free Optimization] <https://epubs.siam.org/doi/book/10.1137/1.9780898718768>`_

The **black-box** nature of many real-world optimization problems comes from one or more of the following factors,
as shown in e.g. the classical book **<<Introduction to Derivative-Free Optimization>>**:

* increasing complexity in mathematical modeling,
* higher sophistication of scientific computing,
* an abundance of legacy or proprietary codes,
* noisy function evaluations.

Some common problem characteristics of BBO are presented below:

* unavailability of gradient information in the black-box setting (even if the gradient information actually exists),
* without a precise model (e.g., owing to complex simulation),
* non-differentiability,
* non-linearity,
* multi-modality,
* ill-condition,
* noisiness.

For BBO, the only information accessible to the algorithm is function evaluations, which can be freely selected by
the algorithm, leading to Zeroth-Order Optimization (ZOO).

No Free Lunch Theorems (NFL)
----------------------------

.. note:: `"In practice it has proven to be crucial to find the right domain-specific trade-off on issues such as
   convergence speed, expected quality of the solutions found and sensitivity to local suboptima on the fitness
   landscape."---[Wierstra et al., 2008] <https://ieeexplore.ieee.org/document/4631255>`_

As mathematically proved in `[Wolpert&Macready, 1997, TEVC] <https://ieeexplore.ieee.org/document/585893>`_, **"for any
algorithm, any elevated performance over one class of problems is offset by performance over another class."**

Curse of Dimensionality for Large-Scale BBO (LSBBO)
---------------------------------------------------

General-Purpose Optimization Algorithms
---------------------------------------

.. note:: *"Given the abundance of black-box optimization algorithms and of optimization problems, how can best match
   algorithms to problems."* ---`[Wolpert&Macready, 1997, TEVC] <https://ieeexplore.ieee.org/document/585893>`_

`"Clearly, evaluating and comparing algorithms on a single problem is not sufficient to determine their quality, as much
of their benefit lies in their performance generalizing to large classes of problems. One of the goals of research in
optimization is, arguably, to provide practitioners with reliable, powerful and general-purpose algorithms."
<https://people.idsia.ch/~schaul/publications/thesis.pdf>`_

The following common criteria/principles may be highly expected to satisfy for general-purpose optimization algorithms:

* effectiveness,
* efficiency,
* elegance (beauty),
* flexibility,
* robustness (reliability),
* scalability,
* simplicity,
* versatility.

Arguably, the *beauty* of general-purpose black-box optimizers should come from **theoretical depth** and/or **practical
breadth**, though the aesthetic judgment is somewhat subjective. We believe that well-designed optimizers could pass
**Test-of-Time**.

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
