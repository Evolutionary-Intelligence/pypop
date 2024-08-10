Black-Box Optimization (BBO)
============================

.. note:: `"Certainly, and especially because of the broad availability of difficult and important applications,
   this promises to be an exciting, interesting, and challenging area for many years to come."---[Conn et al.,
   2009, Introduction to Derivative-Free Optimization, MOS-SIAM Series on Optimization]
   <https://epubs.siam.org/doi/book/10.1137/1.9780898718768>`_

The **black-box** nature of many real-world optimization problems comes from one or more of the following
factors, as shown in the **classical** book `Introduction to Derivative-Free Optimization
<https://epubs.siam.org/doi/book/10.1137/1.9780898718768>`_ or the **seminal** paper `Random Gradient-Free
Minimization of Convex Functions <https://link.springer.com/article/10.1007/s10208-015-9296-2>`_, just to
name a few:

* increasing complexity in mathematical modeling,
* higher sophistication of scientific computing,
* an abundance of legacy or proprietary codes (modification is either too costly or impossible),
* noisy function evaluations,
* memory limitations as fast differentiation needs to all intermediate computations,
* expensive working time (very often working time for computing partial derivatives is much more expensive than
  the computational time),
* a non-trivial extension of the gradient notion onto nonsmooth cases,
* A simple preparatory stage.

.. note:: `"These methods have an evident advantage of a simple preparatory stage (the program of computation of the
   function value is always much simpler than the program for computing the vector of the gradient)."---[Nesterov&Spokoiny,
   2017, FoCM] <https://link.springer.com/article/10.1007/s10208-015-9296-2>`_

Some of common problem characteristics of BBO are presented below:

* Unavailability of the gradient information in various black-box settings, such as

  * `[Moon et al., 2023, Nature Medicine] <https://www.nature.com/articles/s41591-023-02482-6>`_,
    `[Wang et al., 2023, Nature Mental Health] <https://www.nature.com/articles/s44220-023-00110-3>`_,
    `[Xie et al., 2023, Nature Communications] <https://www.nature.com/articles/s41467-023-41951-x>`_,
    `[Mathis et al., 2023, Nature Biotechnology] <https://www.nature.com/articles/s41587-022-01613-7>`_,
    `[Muller et al., 2023, ICML] <https://proceedings.mlr.press/v202/muller23a/muller23a.pdf>`_,
    `[Tian et al., 2023, KDD] <https://dl.acm.org/doi/pdf/10.1145/3580305.3599882>`_,
    `[Schuch et al., 2023, JAMA] <https://jamanetwork.com/journals/jamanetworkopen/article-abstract/2811316>`_,
    `[Cowen-Rivers, 2022, Doctoral Thesis] <https://tuprints.ulb.tu-darmstadt.de/24178/1/Pushing%20The%20Limits%20Of%20Sample-Efficent%20Optimisation.pdf>`_,
    `[Flam-Shepherd et al., 2022, Nature Communications] <https://www.nature.com/articles/s41467-022-30839-x>`_,
    `[Roman et al., 2021, Nature Machine Intelligence] <https://www.nature.com/articles/s42256-021-00312-3>`_,
    `[Beucler et al., 2021, PRL] <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.098302>`_,
    `[Shen et al., 2021, Nature Communications] <https://www.nature.com/articles/s41467-021-26023-2>`_,
    `[Gonatopoulos-Pournatzis et al., 2020, Nature Biotechnology] <https://www.nature.com/articles/s41587-020-0437-z>`_,
    `[Valeri et al., 2020, Nature Communications] <https://www.nature.com/articles/s41467-020-18676-2>`_,
    and so on from the `AutoML <https://www.automl.org/automl/>`_ community;
  * `[Chen et al., 2020, Science Robotics] <https://www.science.org/doi/full/10.1126/scirobotics.abb6938>`_,
    `[Schumer and Steiglitz, 1968, TAC] <https://ieeexplore.ieee.org/abstract/document/1098903>`_,
    `[Karnopp, 1963, Automatica] <https://www.sciencedirect.com/science/article/abs/pii/0005109863900189>`_,
    `[Ashby, 1952] <https://psycnet.apa.org/record/1953-03189-000>`_ from the `Adaptive Control
    <https://www.cds.caltech.edu/archive/help/uploads/wiki/files/140/IEEE_WorkShop_Slides_Lavretsky.pdf>`_ community;
  * `[Brooks, 1959, OR] <https://pubsonline.informs.org/doi/abs/10.1287/opre.7.4.430>`_,
    `[Brooks, 1958, OR] <https://pubsonline.informs.org/doi/10.1287/opre.6.2.244>`_ from the Operations Research (OR) community;
* without a precise mathematical model (e.g., owing to complex simulation), such as

  * `[Pickard&Needs, 2006, PRL] <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.97.045504>`_,
  * `[Robbins, 1952, BAMS] <https://community.ams.org/journals/bull/1952-58-05/S0002-9904-1952-09620-8/S0002-9904-1952-09620-8.pdf>`_.
* non-differentiability, such as

  * when automatic differentiation is not possible (gives noninformative gradients) `[Li et al., 2023, NeurIPS]
    <https://openreview.net/forum?id=VhbV56AJNt>`_;
* non-linearity, such as

  * `nonlinear metamaterials <https://arxiv.org/abs/2307.07606>`_.
* multi-modality/non-convexity, such as

  * `variational quantum eigensolvers <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.033071>`_.
* ill-condition, such as

  * `nonlinear metamaterials <https://arxiv.org/abs/2307.07606>`_.
* noisiness/stochasticity, such as

  * `[Mhanna&Assaad, 2023, ICML] <https://proceedings.mlr.press/v202/mhanna23a/mhanna23a.pdf>`_,
  * `[Bollapragada&Wild, 2023, MPC] <https://link.springer.com/article/10.1007/s12532-023-00233-9>`_,
  * `[Yi et al., 2022, Automatica] <https://www.sciencedirect.com/science/article/pii/S0005109822002035>`_,
  * `[Brooks, 1959, OR] <https://pubsonline.informs.org/doi/abs/10.1287/opre.7.4.430>`_;
* sometimes even `discountinuity`, such as

  * `[Li et al., 2023, NeurIPS] <https://openreview.net/forum?id=VhbV56AJNt>`_.

For black-box problems, the only information accessible to the algorithm is *function evaluations*, which can be freely
selected by the algorithm, leading to Zeroth-Order Optimization (ZOO) or Derivative-Free Optimization (DFO) or
Gradient-Free Optimization (GFO).

No Free Lunch Theorems (NFL)
----------------------------

.. note:: `"In practice it has proven to be crucial to find the right domain-specific trade-off on issues such as
   convergence speed, expected quality of the solutions found and sensitivity to local suboptima on the fitness
   landscape."---[Wierstra et al., 2008] <https://ieeexplore.ieee.org/document/4631255>`_

As mathematically proved in `[Wolpert&Macready, 1997, TEVC] <https://ieeexplore.ieee.org/document/585893>`_, **"for any
algorithm, any elevated performance over one class of problems is offset by performance over another class."**

This may in part explain why there exist a large number of optimization algorithms from different research communities
in practice. However, unfortunately **not** all optimizers are well-designed and widely-acceptable. Refer to the `Design
Philosophy <https://pypop.readthedocs.io/en/latest/design-philosophy.html>`_ section for discussions.

Curse of Dimensionality for Large-Scale BBO (LBO)
-------------------------------------------------

Arguably all black-box optimizers have a possible risk of sufferring from the notorious "Curse of Dimensionality" (also
called `Combinatorial Explosion <https://dl.acm.org/doi/pdf/10.1145/1283920.1283930>`_ in the combinatorial optimization
scenario), since the essence (driving force) of all black-box optimizers are based on **limited sampling** in practice.
Please refer to e.g., `[Nesterov&Spokoiny, 2017, FoCM] <https://link.springer.com/article/10.1007/s10208-015-9296-2>`_
for theoretical analyses.

Luckily, for some real-world applications, there may exist some structures to be available. If such a structure can be
efficiently exploited in an automatic fashion (via well-designed optimization strategies), the convergence rate may be
significantly improved, if possible. Therefore, any general-purpose black-box optimizer may still need to keep a *subtle*
balance between exploiting concrete problem structures and exploring the entire design space of the optimizer.

General-Purpose Optimization Algorithms
---------------------------------------

.. note:: *"Given the abundance of black-box optimization algorithms and of optimization problems, how can best match
   algorithms to problems."*---`[Wolpert&Macready, 1997, TEVC] <https://ieeexplore.ieee.org/document/585893>`_

`"Clearly, evaluating and comparing algorithms on a single problem is not sufficient to determine their quality, as much
of their benefit lies in their performance generalizing to large classes of problems. One of the goals of research in
optimization is, arguably, to provide practitioners with reliable, powerful and general-purpose algorithms."
<https://people.idsia.ch/~schaul/publications/thesis.pdf>`_ As a library for BBO, a natural choice is to first prefer
and cover general-purpose optimization algorithms (when compared with highly-customized versions), since for most
real-world black-box optimization problems the (possibly useful) problem structure is typically unknown in advance.

The following common criteria/principles may be highly expected to satisfy for general-purpose optimization algorithms:

* effectiveness and efficiency,
* elegance (beauty),
* flexibility (versatility),
* robustness (reliability),
* scalability,
* simplicity.

Arguably, the *beauty* of general-purpose black-box optimizers should come from **theoretical depth** and/or **practical
breadth**, though the aesthetic judgment is somewhat *subjective*. We believe that well-designed optimizers could pass
**Test-of-Time** in the history of black-box optimization. For recent critical discussions, refer to e.g.
`"metaphor-based metaheuristics, a call for action: the elephant in the room"
<https://link.springer.com/article/10.1007/s11721-021-00202-9>`_ and `"a critical problem in benchmarking and analysis
of evolutionary computation methods" <https://www.nature.com/articles/s42256-022-00579-0>`_.

For **benchmarking** of continuous optimizers, refer to e.g.
`[Hillstrom, 1977, ACM-TOMS] <https://dl.acm.org/doi/10.1145/355759.355760>`_,
`[More et al., 1981, ACM-TOMS] <https://dl.acm.org/doi/10.1145/355934.355936>`_,
`[Hansen et al., 2021, OMS] <https://www.tandfonline.com/doi/full/10.1080/10556788.2020.1808977>`_,
`[Meunier et al., 2022, TEVC] <https://ieeexplore.ieee.org/abstract/document/9524335>`_. As stated in
`[More et al., 1981, ACM-TOMS]`, "not testing the algorithm on a large number of functions can easily lead to the
cynical observer to conclude that the algorithm was tuned to particular functions".

POPulation-based OPtimization (POP)
-----------------------------------

.. note:: *"The essence of an evolutionary approach to solve a problem is to equate possible solutions to individuals
   in a population, and to introduce a notion of fitness on the basis of solution quality."*---`[Eiben&Smith, 2015,
   Nature] <https://www.nature.com/articles/nature14544>`_

   *"It seems that derivative free algorithms and evolution strategies are totally different algorithms since they
   are motivated from different ideas. However, they are closely related."*---`[Ye&Zhang, 2019]
   <https://arxiv.org/abs/1910.11490>`_

Population-based (particularly evolution- and swarm-based) optimizers (POP) usually have the following advantages for
black-box problems, when particularly compared to individual-based counterparts:

* few *a priori* assumptions (e.g. with a limited knowledge bias),
* flexible framework (easy integration with problem-specific knowledge via e.g. memetic algorithms),
* robust performance (e.g. w.r.t. noisy perturbation or hyper-parameters),
* diverse solutions (e.g. for multi-modal/multi-objective/dynamic optimization),
* novelty (e.g. beyond intuitions for design problems).

For details (models, algorithms, theories, and applications) about POP, please refer to e.g. the following *well-written*
reviews or books (just to name a few):

* Miikkulainen, R. and Forrest, S., 2021. A biological perspective on evolutionary computation. Nature Machine Intelligence, 3(1), pp.9-15.
* Schoenauer, M., 2015. Chapter 28: Evolutionary algorithms. Handbook of Evolutionary Thinking in the Sciences. Springer.
* Eiben, A.E. and Smith, J., 2015. From evolutionary computation to the evolution of things. Nature, 521(7553), pp.476-482.
* De Jong, K.A., Fogel, D.B. and Schwefel, H.P., 1997. A history of evolutionary computation. Handbook of Evolutionary Computation. Oxford University Press.
* Forrest, S., 1993. Genetic algorithms: Principles of natural selection applied to computation. Science, 261(5123), pp.872-878.

For **principled design of continuous stochastic search**, refer to e.g.,
`[Nikolaus&Auger, 2014] <https://link.springer.com/chapter/10.1007/978-3-642-33206-7_8>`_;
`[Wierstra et al., 2014, JMLR] <https://jmlr.org/papers/v15/wierstra14a.html>`_, just to name a few.

For each algorithm family, we try our best to provide some of *wide-recognized* references on its own API documentations.
You can also see `this online project <https://github.com/Evolutionary-Intelligence/DistributedEvolutionaryComputation>`_
for a (growing) paper list of Evolutionary Computation (EC) and Swarm Intelligence (SI) published in many (*though not all*)
*top-tier* and also EC/SI-focused journals and conferences.

Limitations of BBO
------------------

.. note:: *"If you can obtain clean derivatives (even if it requires considerable effort) and the functions defining
   your problem are smooth and free of noise you should not use derivative-free methods."*---`[Conn et al., 2009,
   Introduction to Derivative-Free Optimization] <https://epubs.siam.org/doi/book/10.1137/1.9780898718768>`_

Very importantly, **not all** optimization problems can fit well in black-box optimizers. In fact, their *arbitrary abuses*
in science and engineering have resulted in wide criticism. Although not always, black-box optimizers are often seen as
**"the last choice of search methods"**. Of course, "first-order methods that require knowledge of the gradient are not
always possible in practice." (from `[Mhanna&Assaad, 2023, ICML] <https://proceedings.mlr.press/v202/mhanna23a/mhanna23a.pdf>`_)
