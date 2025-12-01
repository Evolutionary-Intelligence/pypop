Design Philosophy
=================

.. image:: https://visitor-badge.laobi.icu/badge?page_id=Evolutionary-Intelligence.pypop-DesignPhilosophy
   :target: https://visitor-badge.laobi.icu/badge?page_id=Evolutionary-Intelligence.pypop-DesignPhilosophy

As was shown in one `ALJ-2023 <https://tinyurl.com/2sjn8kp9>`_ paper, `"although metaphors can be powerful
inspiration tools, the emergence of hundreds of barely discernible algorithmic variants under different
labels and nomenclatures has been counterproductive to the scientific progress of the field, as it neither
improves our ability to understand and simulate biological systems nor contributes generalizable knowledge
or design principles for global optimization approaches." <https://tinyurl.com/2sjn8kp9>`_

Given a large number of black-box optimizers (BBO) versions/variants which still keep increasing almost every week,
we need some (possibly) widely acceptable criteria to select from them, as presented below in details. For any
**new/missed** BBO in the literature, we have provided an open-access (unified API) interface to help freely add
them, **if necessary**.

Respect for Beauty (Elegance)
-----------------------------

.. note::

   *"If there is a single dominant theme in this ..., it is that practical methods of numerical computation can be
   simultaneously efficient, clever, and --important-- clear."*---Press, W.H., Teukolsky, S.A., Vetterling, W.T. and
   Flannery, B.P., 2007. `Numerical recipes: The art of scientific computing <http://numerical.recipes/>`_.
   Cambridge University Press.

From the *problem-solving* perspective, we empirically prefer to choose the best optimizer for the black-box
optimization problem at hand. For the new problem, however, the best optimizer is often *unknown* in advance
(when without *a prior* knowledge). As a rule of thumb, we need to compare a (often small) set of
available/well-known optimizers and finally choose the best one according to some predefined performance criteria.
From the *academic research* perspective, however, we prefer so-called **beautiful** optimizers, though always
keeping the `No Free Lunch Theorems <https://ieeexplore.ieee.org/document/585893>`_ in mind. Typically, the beauty
of one optimizer comes from the following attractive features: **model novelty (e.g., useful logical concepts and
design frameworks)**, **competitive performance on at least one class of problems**, **theoretical insights (e.g.,
guarantee of global convergence and rate of convergence on some problem classes)**, **clarity/simplicity for
understanding and implementations**, and **well-recognized** `repeatability/reproducibility
<https://www.nature.com/articles/d41586-019-00067-3>`_.

If you find some BBO which is missed in this library to meet the above standard, welcome to launch
`issues <https://github.com/Evolutionary-Intelligence/pypop/issues>`_ or
`pulls <https://github.com/Evolutionary-Intelligence/pypop/pulls>`_. We will consider it to be included in the
*PyPop7* library as soon as possible, if possible. Note that any
`superficial <https://onlinelibrary.wiley.com/doi/full/10.1111/itor.13176>`_
`imitation <https://dl.acm.org/doi/10.1145/3402220.3402221>`_ to well-established optimizers
(i.e., `Old Wine in a New Bottle <https://link.springer.com/article/10.1007/s11721-021-00202-9>`_) will be
**NOT** considered here. Sometimes, several **very complex** optimizers could obtain the top rank on some
competitions consisting of only *artificially-constructed* benchmark functions. However, these optimizers may become
**over-skilled** on these artifacts. In our opinions, a good optimizer should be elegant and `generalizable
<http://incompleteideas.net/IncIdeas/BitterLesson.html>`_. If there is no persuasive/successful applications reported
for it, we will not consider any **very complex** optimizer in this library, in order to avoid the possible `repeatability
<https://dl.acm.org/doi/full/10.1145/3466624>`_ and `overfitting
<http://incompleteideas.net/IncIdeas/BitterLesson.html>`_ issues.

  * Campelo, F. and Aranha, C., 2023. `Lessons from the evolutionary computation Bestiary
    <https://publications.aston.ac.uk/id/eprint/44574/1/ALIFE_LLCS.pdf>`_. Artificial Life, 29(4), pp.421-432.

  * Swan, J., Adriaensen, S., Brownlee, A.E., Hammond, K., Johnson, C.G., Kheiri, A., Krawiec, F., Merelo, J.J.,
    Minku, L.L., Özcan, E., Pappa, G.L., et al., 2022. `Metaheuristics “in the large”
    <https://www.sciencedirect.com/science/article/pii/S0377221721004707>`_. European Journal of Operational Research,
    297(2), pp.393-406.

  * Kudela, J., 2022. `A critical problem in benchmarking and analysis of evolutionary computation methods
    <https://www.nature.com/articles/s42256-022-00579-0>`_. Nature Machine Intelligence, 4(12), pp.1238-1245.

  * Aranha, C., Camacho Villalón, C.L., Campelo, F., Dorigo, M., Ruiz, R., Sevaux, M., Sörensen, K. and Stützle, T., 2022.
    `Metaphor-based metaheuristics, a call for action: The elephant in the room
    <https://link.springer.com/article/10.1007/s11721-021-00202-9>`_. Swarm Intelligence, 16(1), pp.1-6.

  * de Armas, J., Lalla-Ruiz, E., Tilahun, S.L. and Voß, S., 2022. `Similarity in metaheuristics: A gentle step towards a
    comparison methodology <https://link.springer.com/article/10.1007/s11047-020-09837-9>`_. Natural Computing, 21(2),
    pp.265-287.

  * Sörensen, K., Sevaux, M. and Glover, F., 2018. `A history of metaheuristics
    <https://link.springer.com/referenceworkentry/10.1007/978-3-319-07124-4_4>`_. In Handbook of Heuristics (pp. 791-808).
    Springer, Cham.

  * Sörensen, K., 2015. `Metaheuristics—the metaphor exposed <https://onlinelibrary.wiley.com/doi/full/10.1111/itor.12001>`_.
    International Transactions in Operational Research, 22(1), pp.3-18.

  * `Auger, A. <https://scholar.google.com/citations?user=z04BQjgAAAAJ&hl=en&oi=ao>`_, `Hansen, N.
    <https://scholar.google.com/citations?user=Z8ISh-wAAAAJ&hl=en&oi=ao>`_ and `Schoenauer, M.
    <https://scholar.google.com/citations?user=GrCk6WoAAAAJ&hl=en&oi=ao>`_, 2012.
    `Benchmarking of continuous black box optimization algorithms
    <https://direct.mit.edu/evco/article-abstract/20/4/481/956/Benchmarking-of-Continuous-Black-Box-Optimization>`_.
    Evolutionary Computation, 20(4), pp.481-481.

Respect for Diversity
---------------------

Given the universality of **black-box optimization** in science and engineering, different research communities
have designed different kinds of optimizers. The type and number of optimizers are continuing to increase as the more
powerful optimizers are always desirable for new and more challenging applications. On the one hand, some of these
optimizers may share *more or less* similarities. On the other hand, they may also show *significant* differences (w.r.t.
motivations / objectives / implementations / communities / practitioners). Therefore, we hope to cover such a
diversity from different research communities such as artificial intelligence/machine learning (particularly 
`evolutionary computation <https://github.com/Evolutionary-Intelligence/DistributedEvolutionaryComputation>`_, swarm
intelligence, and zeroth-order optimization), mathematical optimization/programming (particularly derivative-free/global
optimization), operations research / management science (`metaheuristics
<https://www.informs.org/Recognizing-Excellence/Award-Recipients/Fred-W.-Glover>`_), automatic control (random search),
electronic engineering, physics, chemistry, open-source software, and many others.

.. note::

   *"The theory of evolution by natural selection explains the adaptedness and diversity of the world solely
   materialistically".*---`[Mayr, 2009, Scientific American]
   <https://www.scientificamerican.com/article/darwins-influence-on-modern-thought1/>`_.

To cover recent advances on population-based BBO as widely as possible, We have actively maintained a `companion project
<https://github.com/Evolutionary-Intelligence/DistributedEvolutionaryComputation>`_ to collect related papers on
some *top-tier* journals and conferences for more than 3 years. We wish that this open companion project could provide an
increasingly reliable literature reference as the base of our library.

.. note::

   `DistributedEvolutionaryComputation <https://github.com/Evolutionary-Intelligence/DistributedEvolutionaryComputation>`_
   provides a (still growing) paper list of Evolutionary Computation (EC) published in some (rather all) top-tier and also
   EC-focused journals and conferences.

Respect for Originality
-----------------------

For each black-box optimizer included in *PyPop7*, we expect to give its original/representative reference (sometimes also
including some of its good implementations/improvements). If you find some important references missed here, please do NOT
hesitate to contact us (and we will be happy to add it). Furthermore, if you identify some mistake regarding originality,
we first apologize for our (possible) mistake and will correct it *timely* within this open-source project. We believe that
the self-correcting power of open source could significantly improve the quality of this library. 

.. note::
  *"It is both enjoyable and educational to hear the ideas directly from the creators".*---Hennessy, J.L. and Patterson,
  D.A., 2019. `Computer architecture: A quantitative approach (Sixth Edition)
  <https://shop.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1>`_. Elsevier.

Respect for Repeatability
-------------------------

For randomized search which is adopted by most population-based optimizers, properly controlling randomness is very
crucial to repeat numerical experiments. Here we follow the official `Random Sampling
<https://numpy.org/doc/stable/reference/random/generator.html>`_ suggestions from `NumPy
<https://numpy.org/doc/stable/reference/random/>`_. In other worlds, you should **explicitly** set the random seed for
each optimizer. For more discussions about **repeatability/benchmarking** from AI/ML, evolutionary computation (EC), swarm
intelligence (SI), and metaheuristics communities, please refer to the following papers, to name a few:

  * López-Ibáñez, M., Paquete, L. and Preuss, M., 2024. `Editorial for the special issue on reproducibility
    <https://direct.mit.edu/evco/article-abstract/32/1/1/119437/Editorial-for-the-Special-Issue-on-Reproducibility>`_.
    Evolutionary Computation, 32(1), pp.1-2.

  * Hansen, N., Auger, A., Brockhoff, D. and Tušar, T., 2022. `Anytime performance assessment in blackbox optimization
    benchmarking <https://ieeexplore.ieee.org/abstract/document/9905722>`_. IEEE Transactions on Evolutionary Computation,
    26(6), pp.1293-1305.

  * Bäck, T., Doerr, C., Sendhoff, B. and Stützle, T., 2022. `Guest editorial special issue on benchmarking sampling-based
    optimization heuristics: Methodology and software <https://ieeexplore.ieee.org/abstract/document/9967395>`_. IEEE
    Transactions on Evolutionary Computation, 26(6), pp.1202-1205.

  * López-Ibáñez, M., Branke, J. and Paquete, L., 2021. `Reproducibility in evolutionary computation
    <https://dl.acm.org/doi/abs/10.1145/3466624>`_. ACM Transactions on Evolutionary Learning and Optimization,
    1(4), pp.1-21.

  * Hutson, M., 2018. `Artificial intelligence faces reproducibility crisis
    <https://www.science.org/doi/10.1126/science.359.6377.725>`_. Science, 359(6377), pp.725-726.

  * Swan, J., Adriaensen, S., Bishr, M., et al., 2015, June. `A research agenda for metaheuristic standardization
    <http://www.cs.nott.ac.uk/~pszeo/docs/publications/research-agenda-metaheuristic.pdf>`_. In Proceedings of International
    Conference on Metaheuristics (pp. 1-3).

  * Sonnenburg, S., Braun, M.L., Ong, C.S., et al., 2007. `The need for open source software in machine learning
    <https://jmlr.csail.mit.edu/papers/volume8/sonnenburg07a/sonnenburg07a.pdf>`_. Journal of Machine Learning Research,
    8, pp.2443-2466.

For benchmarking, please refer to e.g., `BBSR - Benchmarking, Benchmarks, Software, and Reproducibility in ACM GECCO 2025
<https://gecco-2025.sigevo.org/Tracks#BBSR%20-%20Benchmarking,%20Benchmarks,%20Software,%20and%20Reproducibility>`_.

Finally, we expect to see more interesting discussions about BBO from different perspectives. For any **new/missed** BBO,
we provide a *unified* API interface to help freely add them if it satisfies the above design philosophy well. Please
refer to the `Development Guide <https://pypop.readthedocs.io/en/latest/development-guide.html>`_ for details.

* 2025: `The paradox of success in evolutionary and bioinspired optimization: Revisiting
  critical issues, key studies, and methodological pathways
  <https://arxiv.org/abs/2501.07515>`_
* 2024: `Research orientation and novelty discriminant for new metaheuristic algorithms
  <https://www.sciencedirect.com/science/article/pii/S1568494624002953>`_
* 2024: `Metaheuristics exposed: Unmasking the design pitfalls of *** optimization
  algorithm in benchmarking
  <https://www.sciencedirect.com/science/article/abs/pii/S1568494624004708>`_
* 2024: `Comprehensive taxonomies of nature- and bio-inspired optimization: Inspiration
  versus algorithmic behavior, critical analysis and recommendations (from 2020 to 2024)
  <https://arxiv.org/abs/2002.08136>`_
* 2024: `Exposing the *** optimization algorithm: A misleading metaheuristic technique with
  structural bias <https://www.sciencedirect.com/science/article/pii/S156849462400348X>`_
* 2024: `A literature review and critical analysis of metaheuristics recently developed
  <https://link.springer.com/article/10.1007/s11831-023-09975-0>`_
* 2023: `Exposing the ***, ***, ***, ***, ***, and *** algorithms: six misleading
  optimization techniques inspired by bestial metaphors
  <https://onlinelibrary.wiley.com/doi/full/10.1111/itor.13176>`_
* 2022: `A new taxonomy of global optimization algorithms
  <https://link.springer.com/article/10.1007/s11047-020-09820-4>`_
* 2020: `Nature inspired optimization algorithms or simply variations of metaheuristics?
  <https://link.springer.com/article/10.1007/s10462-020-09893-8>`_



.. image:: https://visitor-badge.laobi.icu/badge?page_id=Evolutionary-Intelligence.pypop
   :target: https://visitor-badge.laobi.icu/badge?page_id=Evolutionary-Intelligence.pypop
