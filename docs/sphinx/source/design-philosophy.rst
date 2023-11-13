Design Philosophy of PyPop7
===========================

Given a large number of black-box optimizers (BBO) which keep increasing almost every month, we need some (possibly)
widely acceptable criteria to select from them, as presented below in details:

* Respect for **Beauty (Elegance)**

  From the *problem-solving* perspective, we empirically prefer to choose the best optimizer for the black-box
  optimization problem at hand. For the new problem, however, the best optimizer is often *unknown* in advance
  (when without *a prior* knowledge). As a rule of thumb, we need to compare a (often small) set of
  available/well-known optimizers and finally choose the best one according to some predefined performance criteria.
  From the *academic research* perspective, however, we prefer so-called **beautiful** optimizers, though always
  keeping the `No Free Lunch Theorems <https://ieeexplore.ieee.org/document/585893>`_ in mind. Typically, the beauty
  of one optimizer comes from the following attractive features: **model novelty**, **competitive performance on
  at least one class of problems**, **theoretical insights (e.g., convergence)**, **clarity/simplicity for
  understanding and implementation**, and **repeatability**.

  If you find any to meet the above standard, welcome to launch
  `issues <https://github.com/Evolutionary-Intelligence/pypop/issues>`_ or
  `pulls <https://github.com/Evolutionary-Intelligence/pypop/pulls>`_. We will consider it to be included in the
  *pypop7* library as soon as possible, if possible. Note that any
  `superficial <https://onlinelibrary.wiley.com/doi/full/10.1111/itor.13176>`_
  `imitation <https://dl.acm.org/doi/10.1145/3402220.3402221>`_ to well-established optimizers
  (i.e. `Old Wine in a New Bottle <https://link.springer.com/article/10.1007/s11721-021-00202-9>`_) will be
  **NOT** considered here. Sometimes, several **very complex** optimizers could obtain the top rank on some
  competitions consisting of only *artificially-constructed* benchmark functions. However, these optimizers may become
  **over-skilled** on these artifacts. In our opinions, a good optimizer should be elegant (at least understandable)
  and `generalizable <http://incompleteideas.net/IncIdeas/BitterLesson.html>`_. If there is no persuasive/successful
  real-world applications reported for it, we will not consider any **very complex** optimizer in this library, in order
  to aovid the possible `repeatability <https://dl.acm.org/doi/full/10.1145/3466624>`_ and `overfitting
  <http://incompleteideas.net/IncIdeas/BitterLesson.html>`_ issues.

  * Campelo, F. and Aranha, C., 2023. `Lessons from the evolutionary computation Bestiary
    <https://publications.aston.ac.uk/id/eprint/44574/1/ALIFE_LLCS.pdf>`_. Artificial Life. Early Access.

  * Swan, J., Adriaensen, S., Brownlee, A.E., Hammond, K., Johnson, C.G., Kheiri, A., Krawiec, F., Merelo, J.J.,
    Minku, L.L., Özcan, E., Pappa, G.L., et al., 2022. `Metaheuristics “in the large”
    <https://www.sciencedirect.com/science/article/pii/S0377221721004707>`_. European Journal of Operational Research,
    297(2), pp.393-406.

  * Kudela, J., 2022. `A critical problem in benchmarking and analysis of evolutionary computation methods
    <https://www.nature.com/articles/s42256-022-00579-0>`_. Nature Machine Intelligence, 4(12), pp.1238-1245.

  * Aranha, C., Camacho Villalón, C.L., Campelo, F., Dorigo, M., Ruiz, R., Sevaux, M., Sörensen, K. and Stützle, T., 2022.
    `Metaphor-based metaheuristics, a call for action: The elephant in the room
    <https://link.springer.com/article/10.1007/s11721-021-00202-9>`_. Swarm Intelligence, 16(1), pp.1-6.

  * Piotrowski, A.P. and Napiorkowski, J.J., 2018. `Some metaheuristics should be simplified
    <https://www.sciencedirect.com/science/article/abs/pii/S0020025517310332>`_. Information Sciences, 427, pp.32-62.

  * Sörensen, K., Sevaux, M. and Glover, F., 2018. `A history of metaheuristics
    <https://link.springer.com/referenceworkentry/10.1007/978-3-319-07124-4_4>`_. In Handbook of Heuristics (pp. 791-808).
    Springer, Cham.

  * Sörensen, K., 2015. `Metaheuristics—the metaphor exposed <https://onlinelibrary.wiley.com/doi/full/10.1111/itor.12001>`_.
    International Transactions in Operational Research, 22(1), pp.3-18.

  * Auger, A., Hansen, N. and Schoenauer, M., 2012. `Benchmarking of continuous black box optimization algorithms
    <https://direct.mit.edu/evco/article-abstract/20/4/481/956/Benchmarking-of-Continuous-Black-Box-Optimization>`_.
    Evolutionary Computation, 20(4), pp.481-481.

.. note::

  *"If there is a single dominant theme in this ..., it is that practical methods of numerical computation can be
  simultaneously efficient, clever, and --important-- clear."*---Press, W.H., Teukolsky, S.A., Vetterling, W.T. and
  Flannery, B.P., 2007. `Numerical recipes: The art of scientific computing <http://numerical.recipes/>`_.
  Cambridge University Press.

* Respect for **Diversity**

  Given the universality of **black-box optimization (BBO)** in science and engineering, different research communities
  have designed different optimizers/methods. The type and number of optimizers are continuing to increase as the more
  powerful optimizers are always desirable for new and more challenging applications. On the one hand, some of these
  methods may share *more or less* similarities. On the other hand, they may also show *significant* differences (w.r.t.
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

* Respect for **Originality**

  For each optimizer included in *PyPop7*, we expect to give its original/representative reference (sometimes also
  including its good implementations/improvements). If you find some important references missed, please do NOT hesitate
  to contact us (and we will be happy to add it).

.. note::
  *"It is both enjoyable and educational to hear the ideas directly from the creators".*---Hennessy, J.L. and Patterson,
  D.A., 2019. `Computer architecture: A quantitative approach (Sixth Edition)
  <https://shop.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1>`_. Elsevier.

* Respect for **Repeatability**

  For randomized search which is adopted by most population-based optimizers, properly controlling randomness is very
  crucial to repeat numerical experiments. Here we follow the `Random Sampling
  <https://numpy.org/doc/stable/reference/random/generator.html>`_ suggestions from `NumPy
  <https://numpy.org/doc/stable/reference/random/>`_. In other worlds, you should **explicitly** set the random seed for
  each optimizer. For more discussions about **repeatability** from AI/ML, evolutionary computation, and  metaheuristics
  communities, refer to the following papers, to name a few:

  * López-Ibáñez, M., Branke, J. and Paquete, L., 2021. `Reproducibility in evolutionary computation
    <https://dl.acm.org/doi/abs/10.1145/3466624>`_. ACM Transactions on Evolutionary Learning and Optimization,
    1(4), pp.1-21.

  * Hutson, M., 2018. `Artificial intelligence faces reproducibility crisis
    <https://www.science.org/doi/10.1126/science.359.6377.725>`_. Science, 359(6377), pp.725-726.

  * Sonnenburg, S., Braun, M.L., Ong, C.S., Bengio, S., Bottou, L., Holmes, G., LeCunn, Y., Muller, K.R., Pereira, F.,
    Rasmussen, C.E., Ratsch, G., et al., 2007. `The need for open source software in machine learning
    <https://jmlr.csail.mit.edu/papers/volume8/sonnenburg07a/sonnenburg07a.pdf>`_. Journal of Machine Learning Research,
    8, pp.2443-2466.

We expect to see more interesting discussions about the **Beauty of Black-Box Optimizers**! For **new/missed** BBO, we
provide a *unified* API interface to freely add them if they satisfy the above design philosophy (see
`development-guide <https://pypop.readthedocs.io/en/latest/development-guide.html>`_ for details).
