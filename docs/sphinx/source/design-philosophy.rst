Design Philosophy
=================

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

  * Aranha, C., Camacho Villalón, C.L., Campelo, F., Dorigo, M., Ruiz, R., Sevaux, M., Sörensen, K. and Stützle, T., 2022.
    `Metaphor-based metaheuristics, a call for action: The elephant in the room
    <https://link.springer.com/article/10.1007/s11721-021-00202-9>`_. Swarm Intelligence, 16(1), pp.1-6.

  * Piotrowski, A.P. and Napiorkowski, J.J., 2018. `Some metaheuristics should be simplified
    <https://www.sciencedirect.com/science/article/abs/pii/S0020025517310332>`_. Information Sciences, 427, pp.32-62.

.. note::

  *"If there is a single dominant theme in this ..., it is that practical methods of numerical computation can be
  simultaneously efficient, clever, and --important-- clear."*---Press, W.H., Teukolsky, S.A., Vetterling, W.T. and
  Flannery, B.P., 2007. Numerical recipes: The art of scientific computing. Cambridge University Press.

* Respect for **Diversity**

  Given the universality of **black-box optimization (BBO)** in science and engineering, different research communities
  have designed different optimizers/methods. The type and number of optimizers are continuing to increase as the more
  powerful optimizers are always desirable for new and more challenging applications. On the one hand, some of these
  methods may share *more or less* similarities. On the other hand, they may also show *significant* differences (w.r.t.
  motivations / objectives / implementations / communities / practitioners). Therefore, we hope to cover such a
  diversity from different research communities such as artificial intelligence (particularly machine learning
  (`evolutionary computation <https://github.com/Evolutionary-Intelligence/DistributedEvolutionaryComputation>`_ and
  zeroth-order optimization)), mathematical optimization/programming (particularly global optimization), operations
  research / management science, automatic control, electronic engineering, open-source software, physics, chemistry,
  and others.

* Respect for **Originality**

  For each optimizer included in *PyPop7*, we expect to give its original/representative reference (sometimes also
  including its good implementations/improvements). If you find some important references missed, please do NOT hesitate
  to contact us (and we will be happy to add it if necessary).

.. note::
  *"It is both enjoyable and educational to hear the ideas directly from the creators".*---Hennessy, J.L. and Patterson,
  D.A., 2019. Computer architecture: A quantitative approach (Sixth Edition). Elsevier.

* Respect for **Repeatability**

  For randomized search, properly controlling randomness is very crucial to repeat numerical experiments. Here we follow
  the `Random Sampling <https://numpy.org/doc/stable/reference/random/generator.html>`_ suggestions from `NumPy
  <https://numpy.org/doc/stable/reference/random/>`_. In other worlds, you must **explicitly** set the random seed for
  each optimizer. For more discussions about **repeatability** from machine learning, evolutionary computation, and 
  metaheuristics communities, refer to the following papers, to name a few:
    
  * López-Ibáñez, M., Branke, J. and Paquete, L., 2021. `Reproducibility in evolutionary computation
    <https://dl.acm.org/doi/abs/10.1145/3466624>`_. ACM Transactions on Evolutionary Learning and Optimization,
    1(4), pp.1-21.

  * Sonnenburg, S., Braun, M.L., Ong, C.S., Bengio, S., Bottou, L., Holmes, G., LeCunn, Y., Muller, K.R., Pereira, F.,
    Rasmussen, C.E., Ratsch, G., et al., 2007. `The need for open source software in machine learning
    <https://jmlr.csail.mit.edu/papers/volume8/sonnenburg07a/sonnenburg07a.pdf>`_. Journal of Machine Learning Research,
    8, pp.2443-2466.

We expect to see more interesting discussions about the **Beauty of Black-Box Optimizers (BBO)**!
