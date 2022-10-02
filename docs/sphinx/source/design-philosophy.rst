Design Philosophy of PyPop7
===========================

Given a large number of (black-box) optimizers which keep increasing almost every day, we need some (possibly) widely acceptable criteria to select from them, as presented below in details:

* Respect for **Beauty (Elegance)**

  From the *problem-solving* perspective, we empirically prefer to choose the best optimizer for the black-box optimization problem at hand. For the new problem, however, the best optimizer is often *unknown* in advance (when without *a prior* knowledge). As a rule of thumb, we need to compare a (often small) set of all available/well-known optimizers and finally choose the best one according to some predefined performance criteria. From the *academic research* perspective, however, we prefer so-called **beautiful** optimizers, though always keeping the `No Free Lunch Theorems <https://ieeexplore.ieee.org/document/585893>`_ in mind. Typically, the beauty of one optimizer comes from the following attractive features: **model novelty**, **competitive performance on at least one class of problems**, **theoretical insights (e.g., convergence)**, **clarity/simplicity for understanding and implementation**, and **repeatability**.

  If you find any to meet the above standard, welcome to launch `issues <https://github.com/Evolutionary-Intelligence/pypop/issues>`_ or `pulls <https://github.com/Evolutionary-Intelligence/pypop/pulls>`_. We will consider it to be included in the *pypop7* library as soon as possible, if possible. Note that any `superficial <https://onlinelibrary.wiley.com/doi/full/10.1111/itor.13176>`_ `imitation <https://dl.acm.org/doi/10.1145/3402220.3402221>`_ to well-established optimizers (`Old Wine in a New Bottle <https://link.springer.com/article/10.1007/s11721-021-00202-9>`_) will be **NOT** considered here.

.. note::

  *"If there is a single dominant theme in this ..., it is that practical methods of numerical computation can be simultaneously efficient, clever, and --important-- clear."*

  --- Press, W.H., Teukolsky, S.A., Vetterling, W.T. and Flannery, B.P., 2007. Numerical recipes: The art of scientific computing. Cambridge University Press.

* Respect for **Diversity**

  Given the universality of **black-box optimization (BBO)** in science and engineering, different research communities have designed different optimizers/methods. The type and number of optimizers are continuing to increase as the more powerful optimizers are always desirable for new and more challenging applications. On the one hand, some of these methods may share *more or less* similarities. On the other hand, they may also show *significant* differences (w.r.t. motivations / objectives / implementations / communities / practitioners). Therefore, we hope to cover such a diversity from different research communities such as artificial intelligence (particularly machine learning (evolutionary computation and zeroth-order optimization)), mathematical optimization/programming (particularly global optimization), operations research / management science, automatic control, open-source software, and perhaps others.

* Respect for **Originality**

  For each optimizer included in *pypop7*, we expect to give its original/representative reference (also including its good implementations/improvements). If you find some important references missed, please do NOT hesitate to contact us (and we will be happy to add it if necessary).

.. note::
  *"It is both enjoyable and educational to hear the ideas directly from the creators".*

  --- Hennessy, J.L. and Patterson, D.A., 2019. Computer architecture: A quantitative approach (Sixth Edition). Elsevier.

* Respect for **Repeatability**

  For randomized search, properly controlling randomness is very crucial to repeat numerical experiments. Here we follow the `Random Sampling <https://numpy.org/doc/stable/reference/random/generator.html>`_ suggestions from `NumPy <https://numpy.org/doc/stable/reference/random/>`_. In other worlds, you must **explicitly** set the random seed for each optimizer.

We expect to see more discussions about **Beauty of Black-Box Optimizers (BBO)**!
