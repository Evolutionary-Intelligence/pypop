# PyPop7: A Pure-PYthon librarY of POPulation-based continuous OPtimization in black-box cases [CCF-A]

<img src="https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/logo/MayorIcons.png"
alt="drawing" width="21" height="21"/> `PyPop7` has been used and/or cited in one **Nature**
paper [[Veenstra et al., Nature, 2025]](https://www.nature.com/articles/s41586-025-08646-3) and
etc. For any questions or helps, please directly use
[Discussions](https://github.com/Evolutionary-Intelligence/pypop/discussions).

[![Pub](https://img.shields.io/badge/JMLR-2024-red)](https://jmlr.org/papers/v25/23-0386.html)
[![arXiv](https://img.shields.io/badge/arXiv-2022-orange)](https://arxiv.org/abs/2212.05652)
[![Jan26-2021](https://img.shields.io/badge/Jan26-2021-blue)](https://github.com/Evolutionary-Intelligence/pypop/commit/d99805746409d4e6d841dc7729d8eb3463e97e50)
[![PyPI](https://tinyurl.com/murf7c4m)](https://pypi.org/project/pypop7/)
[![Docs](https://readthedocs.org/projects/pypop/badge/?version=latest)](http://pypop.rtfd.io/)
[![Downloads](https://static.pepy.tech/badge/pypop7)](https://pepy.tech/project/pypop7)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=Evolutionary-Intelligence.pypop)
[![Python](https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Linux](https://img.shields.io/badge/Linux-FCC624?&logo=linux&logoColor=black)](https://www.linuxfoundation.org/)
[![Apple](https://shields.io/badge/MacOS--9cf?logo=Apple&style=social)](https://www.apple.com/macos)
[![Windows](https://custom-icon-badges.demolab.com/badge/Windows-0078D6?logo=windows11&logoColor=white)](https://www.microsoft.com/en-us/windows)
[![CircleCI](https://img.shields.io/badge/CircleCI-343434?logo=circleci&logoColor=fff)](https://app.circleci.com/pipelines/github/Evolutionary-Intelligence/pypop)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=github-actions&logoColor=white)](https://github.com/Evolutionary-Intelligence/pypop/actions)
[![Codecov](https://img.shields.io/badge/Codecov-F01F7A?logo=codecov&logoColor=fff)](https://coverage.readthedocs.io/)
[![PyCharm](https://img.shields.io/badge/PyCharm-000?logo=pycharm&logoColor=fff)](https://www.jetbrains.com/pycharm/)
[![PyTest](https://img.shields.io/badge/pytest-%23ffffff.svg?logo=PyTest&logoColor=2f9fe3)](https://docs.pytest.org/en/stable/)
[![Awesome](https://awesome.re/badge.svg)](https://github.com/Evolutionary-Intelligence/DistributedEvolutionaryComputation)
[![Sphinx](https://img.shields.io/badge/Sphinx-000?logo=sphinx&logoColor=fff)](https://www.sphinx-doc.org/en/master/)
[![WeChat](https://img.shields.io/badge/WeChat-07C160?logo=wechat&logoColor=white)](https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/WeChat/WeChat-20250906.jpg)
[![EvoI](https://img.shields.io/badge/CCF-A-brown)](https://evolutionary-intelligence.github.io)

```PyPop7``` is a [Python](https://www.python.org/) library of **population-based
randomized optimization algorithms** for **single-objective**, **real-parameter**,
**unconstrained** *black-box optimization* (BBO) problems. Its main design goal is
to provide a *unified* interface and a large set of *elegant* implementations for
e.g., [evolutionary algorithms](https://www.nature.com/articles/nature14544),
[swarm optimizers](https://github.com/Evolutionary-Intelligence/SwarmIntelligence-A-Modern-Perspective-SIAMP),
and [pattern search](https://epubs.siam.org/doi/abs/10.1137/S1052623493250780),
with at least three functionalities: (1) To facilitate **research repeatability**
in a controllable manner, (2) To promote **wide benchmarking** in an open-source
fashion, especially (3) To be used in **real-world BBO applications** in a
trial-and-error manner.

Specifically speaking, for alleviating the notorious **curse-of-dimensionality**
issue on BBO, its focus is to cover **State Of The Art (SOTA) for Large-Scale
Optimization** only under Black-Box scenarios, though many of small- and
medium-scaled algorithm versions or variants are also included here (mainly for
**theoretical** or **benchmarking** or **educational** or **practical** purposes).
For a *growing* list of its *diverse* use and/or citation cases, please refer to
[this online website](https://pypop.readthedocs.io/en/latest/applications.html).
Although we have chosen *GPL-3.0 license* initially, anyone could use, modify,
and improve it **entirely freely** for any (no matter *open-source* or
*closed-source*) **positive** purposes.

## Quickstart

"[In our opinion, the main fact, which should be known to any person dealing with optimization
models, is that in general, optimization problems are unsolvable. This statement, which is
usually missing in standard optimization courses, is very important for understanding
optimization theory and the logic of its development in the past and in the
future.](https://tinyurl.com/4yccr5k8)"---From
**[Prof. Yurii Nesterov (Member of National Academy of Sciences,
USA)](https://nrc88.nas.edu/pnas_search/memberDetails.aspx?ctID=20022958)**

The following **3** simple steps are enough to utilize the **black-box optimization**
power of [PyPop7](https://pypi.org/project/pypop7/):

* Recommend using [pip](https://pypi.org/project/pip/) to install ```pypop7``` on the
  Python3-based virtual environment via [venv](https://docs.python.org/3/library/venv.html)
  or [conda](https://docs.conda.io/projects/conda/en/stable/) [*but not mandatory*]:

```bash
$ pip install pypop7
```

For using **free** Miniconda to download and install the virtual environment of Python,
please refer to https://www.anaconda.com/docs/getting-started/miniconda/main.

For ```PyPop7```, the number ```7``` was added just because ```pypop``` has been registered
by [other](http://pypop.org/) in [PyPI](https://pypi.org/). Its icon *butterfly* is used to
respect/allude to the great book (butterflies in its cover) of **Fisher** (["(one of) the
greatest of Darwin's successors"](https://link.springer.com/article/10.1007/s00265-010-1122-x)):
[The Genetical Theory of Natural Selection](https://tinyurl.com/3we44pt4), which directly
inspired [Prof. Holland](https://cacm.acm.org/research/adaptive-computation/)'s
[proposal](https://direct.mit.edu/books/edited-volume/3809/chapter-abstract/125036/An-Interview-with-John-Holland)
of [Genetic Algorithms (GA)](https://dl.acm.org/doi/10.1145/321127.321128).

* Define the objective (cost / loss / error / fitness) function to be **minimized** for
  the optimization problem at hand (here the term **fitness function** is used, in order
  to follow the *established* terminology tradition in **evolutionary computation**):

```Python
import numpy as np  # for numerical computation (as the computing engine of pypop7)

# Rosenbrock (one notorious test function from the continuous optimization community)
def rosenbrock(x):
    return 100.0 * np.sum(np.square(x[1:] - np.square(x[:-1]))) + np.sum(np.square(x[:-1] - 1.0))

# to define the fitness function to be *minimized* and also its problem settings (`dict`)
ndim_problem = 1000
problem = {'fitness_function': rosenbrock,  # fitness function corresponding to the problem
           'ndim_problem': ndim_problem,  # number of dimension of the problem to be optimized
           'lower_boundary': -5.0 * np.ones((ndim_problem,)),  # lower boundary of search range
           'upper_boundary': 5.0 * np.ones((ndim_problem,))}  # upper boundary of search range
```

Without loss of generality, only the **minimization** process is considered, since
**maximization** can be easily transferred to **minimization** just by negating (-) it.

* Run one black-box optimizer (or more) on the above optimization problem. Owing to its
  *low* computational complexity and *well* metric-learning ability, choose **LM-MA-ES**
  as one example. Please refer to https://pypop.readthedocs.io/en/latest/es/lmmaes.html
  for its algorithmic procedure in detail.

```Python
# LMMAES: Limited Memory Matrix Adaptation Evolution Strategy
from pypop7.optimizers.es.lmmaes import LMMAES
# to define algorithm options (which differ in details among different optimizers)
options = {'fitness_threshold': 1e-10,  # to stop if best-so-far fitness <= 1e-10
           'max_runtime': 3600.0,  # to stop if runtime >= 1 hours (3600 seconds)
           'seed_rng': 0,  # random seed (which should be set for repeatability)
           'x': 4.0 * np.ones((ndim_problem,)),  # mean of search distribution
           'sigma': 3.0,  # global step-size (but not necessarily optimal)
           'verbose': 500}
lmmaes = LMMAES(problem, options)  # to initialize (under a unified API)
results = lmmaes.optimize()  # to run its (time-consuming) evolution process
print(results)
```

Clearly, to obtain the *nearly optimal* rate of convergence for ```LMMAES```,
one *key* hyper-parameter ```sigma``` often needs to be well fine-tuned for
this popular test function ```rosenbrock```. In practice, **Hyper-Parameter
Optimization (HPO)** is one very common strategy to **approximate** the
*possibly best* solution for the *complex* optimization problem at hand.
Please refer to e.g., the following books and papers as some (rather all)
representative *reference*:

* [Hutter, F.](), et al., 2019.
  [Automated machine learning: Methods, systems,
  challenges](https://www.automl.org/wp-content/uploads/2019/05/AutoML_Book.pdf).
  Springer.
* Bergstra, J. and [Bengio, Y.](), 2012.
  [Random search for hyper-parameter
  optimization](https://www.jmlr.org/papers/v13/bergstra12a.html).
  JMLR, 3(10), pp.281-305.
* [Hoos, H.H.](), 2011.
  [Automated algorithm configuration and parameter
  tuning](https://link.springer.com/chapter/10.1007/978-3-642-21434-9_3).
  In Autonomous Search (pp. 37-71). Springer.

### Online Documentations, Online Tutorials, and Future Extensions

Please refer to [https://pypop.rtfd.io/](https://pypop.rtfd.io/) for online
documentations and tutorials of this *well-designed* ("**self-boasted**" by
ourselves) Python library for Black-Box Optimization (e.g., [online praises
from others](https://pypop.readthedocs.io/en/latest/applications.html)). A
total of **4** extended versions of PyPop7 (as **PP7**) are *ongoing* or
*planned* for further development:

* For Constrained Black-Box Optimization (`PyCoPop7 as PCP7`),
* For Noisy Black-Box Optimization (`PyNoPop7 as PNP7`),
* Enhancement via Parallel and Distributed Optimization (`PyDPop7 as PDP7`),
* Enhancement via Meta-evolution based Optimization (`PyMePop7 as PMP7`).

## Black-Box Optimizers (BBO)

* "[The main lesson of the development of our field in the last few decades is that efficient
optimization methods can be developed only by intelligently employing the structure of
particular instances of problems.](https://link.springer.com/book/10.1007/978-3-319-91578-4)"
---From **BOOK 'Lectures on Convex Optimization' of Prof. Yurii Nesterov (Member of National
Academy of Sciences, USA) in [Springer-2018]**
* "[Optimization algorithms are often designed for a specific type of search space, exploiting
its specific structure.](https://www.jmlr.org/papers/volume18/14-467/14-467.pdf)"
---From **PAPER 'Information-Geometric Optimization Algorithms: A Unifying Picture via
Invariance Principles' of Nikolaus Hansen (Inventor of CMA-ES) et al in [JMLR-2017]**

******* *** ******* ******* *** ******* ******* *** ******* ******* *** *******
The below algorithm classification based on *only* the dimensionality of objective
function, is *just a roughly empirical* estimation for the basic [algorithm
selection](https://tinyurl.com/ae64wyj3) task. In practice, perhaps the **simplest**
way to algorithm selection is **trial-and-error**. More advanced [automated
algorithm selection](https://doi.org/10.1162/evco_a_00242) techniques can be also
considered here in principle.

* ![lso](https://img.shields.io/badge/*-l-orange.svg): indicates the *specific*
  version for **Large-Scale Optimization (LSO)**, e.g., dimension >> 100 (but this
  is not an *absolutely deterministic* number to distinguish LSO).
* ![c](https://img.shields.io/badge/*-c-blue.svg): indicates the **competitive** or **de
  facto** BBO version for *low- or medium-dimensional* problems (though it may also work
  well under some certain LSO circumstances).
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg): indicates the **baseline** BBO
  version mainly for *theoretical* and/or *educational* interest, owing to its algorithmic
  simplicity (relative ease for mathematical analysis).
******* *** ******* ******* *** ******* ******* *** ******* ******* *** *******

This is an **algorithm-centric** rather than *benchmarking-centric* Python library
(though undoubtedly **proper benchmarking** is crucial for BBO: Via e.g.,
[COCO](https://github.com/numbbo/coco),
[NeverGrad](https://github.com/facebookresearch/nevergrad),
[IOHprofiler](https://iohprofiler.github.io/)).

### Evolution Strategies (ES)

For ```ES```, please refer to e.g.,
[[JMLR-2017]](https://www.jmlr.org/papers/v18/14-467.html),
[[Hansen et al., 2015]](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44),
[[Bäck et al., 2013]](https://link.springer.com/book/10.1007/978-3-642-40137-4),
[[Rudolph, 2012]](https://link.springer.com/referenceworkentry/10.1007/978-3-540-92910-9_22),
[[Beyer&Schwefel, 2002]](https://link.springer.com/article/10.1023/A:1015059928466),
[[Rechenberg, 1989]](https://link.springer.com/chapter/10.1007/978-3-642-83814-9_6),
[[Schwefel, 1984]](https://link.springer.com/article/10.1007/BF01876146),
etc.

* ![lso](https://img.shields.io/badge/*-l-orange.svg)
  Limited-Memory Matrix Adaptation ES (**LMMAES**)
  [[[TEVC-2019]](https://ieeexplore.ieee.org/abstract/document/8410043),
  [[ECJ-2017]](https://direct.mit.edu/evco/article-abstract/25/1/143/1041/LM-CMA-An-Alternative-to-L-BFGS-for-Large-Scale),
  [[GECCO-2014]](https://dl.acm.org/doi/abs/10.1145/2576768.2598294)]
* ![lso](https://img.shields.io/badge/*-l-orange.svg)
  Rank-M ES (**RMES**)
  [[[TEVC-2018]](https://ieeexplore.ieee.org/document/8080257),
  [[PPSN-2016]](https://link.springer.com/chapter/10.1007/978-3-319-45823-6_70)]
* ![lso](https://img.shields.io/badge/*-l-orange.svg)
  Cholesky Covariance Matrix Adaptation ES (**CCMAES2016**)
  [[[NeurIPS-2016]](https://proceedings.neurips.cc/paper/2016/hash/289dff07669d7a23de0ef88d2f7129e7-Abstract.html),
  [[FOGA-2015]](https://dl.acm.org/doi/abs/10.1145/2725494.2725496),
  [[GECCO-2010]](https://dl.acm.org/doi/abs/10.1145/1830483.1830556),
  [[MLJ-2009]](https://link.springer.com/article/10.1007/s10994-009-5102-1),
  [[GECCO-2006]](https://dl.acm.org/doi/abs/10.1145/1143997.1144082)]
* ![lso](https://img.shields.io/badge/*-l-orange.svg)
  Separable CMA-ES (**SEPCMAES**)
  [[[ECJ-2020]](https://direct.mit.edu/evco/article/28/3/405/94999/Diagonal-Acceleration-for-Covariance-Matrix),
  [[Bäck et al., 2013]](https://link.springer.com/book/10.1007/978-3-642-40137-4),
  [[PPSN-2008]](https://link.springer.com/chapter/10.1007/978-3-540-87700-4_30)]
* ![c](https://img.shields.io/badge/*-c-blue.svg)
  Fast MAES (**FMAES**)
  [[[GECCO-2020]](https://dl.acm.org/doi/abs/10.1145/3377929.3389870),
  [[TEVC-2019]](https://ieeexplore.ieee.org/abstract/document/8410043),
  [[TEVC-2017]](https://ieeexplore.ieee.org/abstract/document/7875115/)]
* ![c](https://img.shields.io/badge/*-c-blue.svg)
  **CMAES**
  [[[arXiv-2023/2016]](https://arxiv.org/abs/1604.00772),
  [[ECJ-2003]](https://direct.mit.edu/evco/article-abstract/11/1/1/1139/Reducing-the-Time-Complexity-of-the-Derandomized),
  [[ECJ-2001]](https://direct.mit.edu/evco/article-abstract/9/2/159/892/Completely-Derandomized-Self-Adaptation-in),
  [[CEC-1996]](https://ieeexplore.ieee.org/abstract/document/542381)]

### Natural ES (NES)

For ```NES```, please refer to e.g.,
[[JMLR-2024]](https://www.jmlr.org/papers/v25/22-0564.html),
[[JMLR-2014]](https://jmlr.org/papers/v15/wierstra14a.html),
[[ICML-2009]](https://dl.acm.org/doi/abs/10.1145/1553374.1553522),
[[CEC-2008]](https://ieeexplore.ieee.org/document/4631255),
etc.

* ![lso](https://img.shields.io/badge/*-l-orange.svg)
  Projection-based CMA (**VKDCMA**)
  [[[PPSN-2016]](https://link.springer.com/chapter/10.1007/978-3-319-45823-6_1),
  [[GECCO-2016]](https://dl.acm.org/doi/abs/10.1145/2908812.2908863)]
* ![lso](https://img.shields.io/badge/*-l-orange.svg)
  Linear CMA
  (**VDCMA**)
  [[[GECCO-2014]](https://dl.acm.org/doi/abs/10.1145/2576768.2598258)]
* ![lso](https://img.shields.io/badge/*-l-orange.svg)
  Rank-1 NES
  (**R1NES**)
  [[[GECCO-2013]](https://dl.acm.org/doi/abs/10.1145/2464576.2464608)]
* ![c](https://img.shields.io/badge/*-c-blue.svg)
  Exponential NES
  (**XNES**)
  [[[GECCO-2010]](https://dl.acm.org/doi/abs/10.1145/1830483.1830557)]
* ![c](https://img.shields.io/badge/*-c-blue.svg)
  Exact NES
  (**ENES**)
  [[[ICML-2009]](https://dl.acm.org/doi/abs/10.1145/1553374.1553522)]

### Estimation of Distribution Algorithms (EDA)

For ```EDA```, please refer to e.g.,
[[GECCO-2020]](https://dl.acm.org/doi/abs/10.1145/3377929.3389938),
[[Larrañaga&Lozano, 2002]](https://link.springer.com/book/10.1007/978-1-4615-1539-5),
[[COA-2002]](https://link.springer.com/article/10.1023/A:1013500812258),
[[PPSN-1996]](https://link.springer.com/chapter/10.1007/3-540-61723-X_982),
[[ICML-1995]](https://www.sciencedirect.com/science/article/pii/B9781558603776500141),
etc.

* ![lso](https://img.shields.io/badge/***-lso-orange.svg) Random-Projection EDA
  (**RPEDA**)
  [[Kabán et al., 2016, ECJ](https://direct.mit.edu/evco/article-abstract/24/2/255/1016/Toward-Large-Scale-Continuous-EDA-A-Random-Matrix)]
* ![lso](https://img.shields.io/badge/***-lso-orange.svg) Univariate Marginal Distribution Algorithm
  (**UMDA**)
  [[Larrañaga&Lozano, 2002](https://link.springer.com/book/10.1007/978-1-4615-1539-5),
  [Mühlenbein, 1997, ECJ](https://tinyurl.com/yt78c786)]

### Cross-Entropy Method (CEM)

For ```CEM```, please refer to e.g.,
[Rubinstein&Kroese, 2016](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118631980),
[Hu et al., 2007, OR](https://pubsonline.informs.org/doi/abs/10.1287/opre.1060.0367),
[Kroese et al., 2006, MCAP](https://link.springer.com/article/10.1007/s11009-006-9753-0),
[De Boer et al., 2005, AOR](https://link.springer.com/article/10.1007/s10479-005-5724-z),
[Rubinstein&Kroese, 2004](https://link.springer.com/book/10.1007/978-1-4757-4321-0)],
etc.

* ![lso](https://img.shields.io/badge/***-lso-orange.svg) Differentiable CEM
  (**DCEM**)
  [[Amos&Yarats, 2020, ICML](https://proceedings.mlr.press/v119/amos20a.html)]
* ![c](https://img.shields.io/badge/**-c-blue.svg) Dynamic-Smoothing CEM
  (**DSCEM**)
  [[Kroese et al., 2006, MCAP](https://link.springer.com/article/10.1007/s11009-006-9753-0)]
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) Standard CEM
  (**SCEM**)
  [e.g. [Kroese et al., 2006, MCAP](https://link.springer.com/article/10.1007/s11009-006-9753-0)]

### Differential Evolution (DE)

For ```EDA```, please refer to e.g.,
[Price, 2013](https://link.springer.com/chapter/10.1007/978-3-642-30504-7_8);
[Price et al., 2005](https://link.springer.com/book/10.1007/3-540-31306-0);
[Storn&Price, 1997, JGO](https://link.springer.com/article/10.1023/A:1008202821328)],
etc.

* ![lso](https://img.shields.io/badge/***-lso-orange.svg) Success-History based Adaptive DE
  (**SHADE**) [[Tanabe&Fukunaga, 2013, CEC](https://ieeexplore.ieee.org/document/6557555)]
* ![lso](https://img.shields.io/badge/***-lso-orange.svg) Adaptive DE (**[JADE](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/de/jade.py)**) [[Zhang&Sanderson, 2009, TEVC](https://doi.org/10.1109/TEVC.2009.2014613)]
* ![c](https://img.shields.io/badge/**-c-blue.svg) Composite DE (**[CODE](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/de/code.py)**) [[Wang et al., 2011, TEVC](https://doi.org/10.1109/TEVC.2010.2087271)]
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) Trigonometric-mutation DE (**[TDE](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/de/tde.py)**) [[Fan&Lampinen, 2003, JGO](https://link.springer.com/article/10.1023/A:1024653025686)]
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) Classic DE (**[CDE](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/de/cde.py)**) [e.g. [Storn&Price, 1997, JGO](https://link.springer.com/article/10.1023/A:1008202821328)]

### Particle Swarm Optimizer (PSO)

For ```PSO```, please refer to e.g.,
[Fornasier et al., 2021, JMLR](https://jmlr.csail.mit.edu/papers/v22/21-0259.html);
[Bonyadi&Michalewicz, 2017, ECJ](https://direct.mit.edu/evco/article-abstract/25/1/1/1040/Particle-Swarm-Optimization-for-Single-Objective);
[Rahmat-Samii et al., 2012, PIEEE](https://ieeexplore.ieee.org/document/6204306);
[Escalante et al., 2009, JMLR](https://www.jmlr.org/papers/v10/escalante09a.html);
[Dorigo et al., 2008, Scholarpedia](http://www.scholarpedia.org/article/Particle_swarm_optimization);
[Poli et al., 2007, SI](https://link.springer.com/article/10.1007/s11721-007-0002-0);
[Shi&Eberhart, 1998, CEC](https://ieeexplore.ieee.org/abstract/document/699146);
[Kennedy&Eberhart, 1995, ICNN](https://ieeexplore.ieee.org/document/488968)],
etc.

  * ![lso](https://img.shields.io/badge/***-lso-orange.svg) Cooperative Coevolving PSO (**[CCPSO2](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/pso/ccpso2.py)**) [[Li&Yao, 2012, TEVC](https://ieeexplore.ieee.org/document/5910380/)]
  * ![lso](https://img.shields.io/badge/***-lso-orange.svg) Incremental PSO (**[IPSO](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/pso/ipso.py)**) [[de Oca et al., 2011, TSMCB](https://ieeexplore.ieee.org/document/5582312)]
  * ![lso](https://img.shields.io/badge/***-lso-orange.svg) Cooperative PSO (**[CPSO](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/pso/cpso.py)**) [[Van den Bergh&Engelbrecht, 2004, TEVC](https://ieeexplore.ieee.org/document/1304845)]
  * ![c](https://img.shields.io/badge/**-c-blue.svg) Comprehensive Learning PSO (**[CLPSO](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/pso/clpso.py)**) [[Liang et al., 2006, TEVC](https://ieeexplore.ieee.org/abstract/document/1637688)]
  * ![b](https://img.shields.io/badge/*-b-lightgrey.svg) Standard PSO with a Local (ring) topology (**[SPSOL](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/pso/spsol.py)**) [e.g. [Shi&Eberhart, 1998, CEC](https://ieeexplore.ieee.org/abstract/document/699146); [Kennedy&Eberhart, 1995, ICNN](https://ieeexplore.ieee.org/document/488968)]
  * ![b](https://img.shields.io/badge/*-b-lightgrey.svg) Standard PSO with a global topology (**[SPSO](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/pso/spso.py)**) [e.g. [Shi&Eberhart, 1998, CEC](https://ieeexplore.ieee.org/abstract/document/699146); [Kennedy&Eberhart, 1995, ICNN](https://ieeexplore.ieee.org/document/488968)]

### Cooperative Co-evolution (CC)

For ```CC```, please refer to e.g.,
[Gomez et al., 2008, JMLR](https://jmlr.org/papers/v9/gomez08a.html);
[Panait et al., 2008, JMLR](https://www.jmlr.org/papers/v9/panait08a.html);
[Moriarty&Miikkulainen, 1995, ICML](https://www.sciencedirect.com/science/article/pii/B9781558603776500566);
[Potter&De Jong, 1994, PPSN](https://link.springer.com/chapter/10.1007/3-540-58484-6_269),
etc.

* ![lso](https://img.shields.io/badge/***-lso-orange.svg) Hierarchical CC (**[HCC](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/cc/hcc.py)**) [[Mei et al., 2016, ACM-TOMS](https://dl.acm.org/doi/10.1145/2791291); [Gomez&Schmidhuber, 2005, ACM-GECCO](https://dl.acm.org/doi/10.1145/1068009.1068092)]
* ![lso](https://img.shields.io/badge/***-lso-orange.svg) CoOperative CMA (**[COCMA](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/cc/cocma.py)**) [[Mei et al., 2016, ACM-TOMS](https://dl.acm.org/doi/10.1145/2791291); [Potter&De Jong, 1994, PPSN](https://link.springer.com/chapter/10.1007/3-540-58484-6_269)]
* ![c](https://img.shields.io/badge/**-c-blue.svg) CoOperative co-Evolutionary Algorithm (**[COEA](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/cc/coea.py)**) [e.g. [Panait et al., 2008, JMLR](https://www.jmlr.org/papers/v9/panait08a.html); [Potter&De Jong, 1994, PPSN](https://link.springer.com/chapter/10.1007/3-540-58484-6_269)]
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) CoOperative SYnapse NeuroEvolution (**[COSYNE](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/cc/cosyne.py)**) [[Gomez et al., 2008, JMLR](https://jmlr.org/papers/v9/gomez08a.html); [Moriarty&Miikkulainen, 1995, ICML](https://www.sciencedirect.com/science/article/pii/B9781558603776500566)]

### Simulated Annealing (SA)

For ```SA```, please refer to e.g.,
[Bertsimas&Tsitsiklis, 1993, Statistical Science](https://tinyurl.com/yknunnpt);
[Kirkpatrick et al., 1983, Science](https://www.science.org/doi/10.1126/science.220.4598.671);
[Hastings, 1970, Biometrika](https://academic.oup.com/biomet/article/57/1/97/284580);
[Metropolis et al., 1953, JCP](https://aip.scitation.org/doi/abs/10.1063/1.1699114)],
etc.

* ![lso](https://img.shields.io/badge/***-lso-orange.svg) Enhanced SA (**[ESA](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/sa/esa.py)**) [[Siarry et al., 1997, TOMS](https://dl.acm.org/doi/abs/10.1145/264029.264043)]
* ![c](https://img.shields.io/badge/**-c-blue.svg) Corana et al.' SA (**[CSA](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/sa/csa.py)**) [[Corana et al., 1987, TOMS](https://dl.acm.org/doi/abs/10.1145/29380.29864)]
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) Noisy SA (**[NSA](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/sa/nsa.py)**) [[Bouttier&Gavra, 2019, JMLR](https://www.jmlr.org/papers/v20/16-588.html)]

### Genetic Algorithms ([GA](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ga/ga.py))

For ```GA```, please refer to e.g.,
[Forrest, 1993, Science](https://www.science.org/doi/abs/10.1126/science.8346439);
[Holland, 1973, SICOMP](https://epubs.siam.org/doi/10.1137/0202009);
[Holland, 1962, JACM](https://dl.acm.org/doi/10.1145/321127.321128)],
etc.

* ![lso](https://img.shields.io/badge/***-lso-orange.svg) Global and Local genetic algorithm (**[GL25](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ga/gl25.py)**) [[García-Martínez et al., 2008, EJOR](https://www.sciencedirect.com/science/article/abs/pii/S0377221706006308)]
* ![c](https://img.shields.io/badge/**-c-blue.svg) Generalized Generation Gap with Parent-Centric Recombination (**[G3PCX](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ga/g3pcx.py)**) [[Deb et al., 2002, ECJ](https://direct.mit.edu/evco/article-abstract/10/4/371/1136/A-Computationally-Efficient-Evolutionary-Algorithm)]
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) GENetic ImplemenTOR (**[GENITOR](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ga/genitor.py)**) [e.g. [Whitley et al., 1993, MLJ](https://link.springer.com/article/10.1023/A:1022674030396)]

* **Evolutionary Programming ([EP](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ep/ep.py))** [e.g., [Yao et al., 1999, TEVC](https://ieeexplore.ieee.org/abstract/document/771163); [Fogel, 1994, Statistics and Computing](https://link.springer.com/article/10.1007/BF00175356)]
  * ![lso](https://img.shields.io/badge/***-lso-orange.svg) Lévy distribution based EP (**[LEP](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ep/lep.py)**) [[Lee&Yao, 2004, TEVC](https://ieeexplore.ieee.org/document/1266370)]
  * ![c](https://img.shields.io/badge/**-c-blue.svg) Fast EP (**[FEP](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ep/fep.py)**) [[Yao et al., 1999, TEVC](https://ieeexplore.ieee.org/abstract/document/771163)]
  * ![b](https://img.shields.io/badge/*-b-lightgrey.svg) Classical EP (**[CEP](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ep/cep.py)**) [e.g. [Yao et al., 1999, TEVC](https://ieeexplore.ieee.org/abstract/document/771163); [Bäck&Schwefel, 1993, ECJ](https://direct.mit.edu/evco/article-abstract/1/1/1/1092/An-Overview-of-Evolutionary-Algorithms-for)]
* **Direct Search ([DS](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ds/ds.py))** [e.g. [Powell, 1998, Acta-Numerica](https://www.cambridge.org/core/journals/acta-numerica/article/abs/direct-search-algorithms-for-optimization-calculations/23FA5B19EAF122E02D3724DB1841238C); [Wright, 1996](https://nyuscholars.nyu.edu/en/publications/direct-search-methods-once-scorned-now-respectable); [Hooke&Jeeves, 1961, JACM](https://dl.acm.org/doi/10.1145/321062.321069)]
  * ![c](https://img.shields.io/badge/**-c-blue.svg) Powell's search method (**[POWELL](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ds/powell.py)**) [[SciPy]( https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html); [Powell, 1964, Computer](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ds/powell.py)]
  * ![b](https://img.shields.io/badge/*-b-lightgrey.svg) Generalized Pattern Search (**[GPS](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ds/gps.py)**) [[Kochenderfer&Wheeler, 2019](https://algorithmsbook.com/optimization/files/chapter-7.pdf); [Torczon, 1997, SIAM-JO](https://epubs.siam.org/doi/abs/10.1137/S1052623493250780)]
  * ![b](https://img.shields.io/badge/*-b-lightgrey.svg) Nelder-Mead simplex method (**[NM](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ds/nm.py)**) [[Dean et al., 1975, Science](https://www.science.org/doi/10.1126/science.189.4205.805); [Nelder&Mead, 1965, Computer](https://academic.oup.com/comjnl/article-abstract/7/4/308/354237)]
  * ![b](https://img.shields.io/badge/*-b-lightgrey.svg) Hooke-Jeeves direct search method (**[HJ](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ds/hj.py)**) [[Kochenderfer&Wheeler, 2019](https://algorithmsbook.com/optimization/files/chapter-7.pdf); [Kaupe, 1963, CACM](https://dl.acm.org/doi/pdf/10.1145/366604.366632); [Hooke&Jeeves, 1961, JACM](https://dl.acm.org/doi/10.1145/321062.321069)]
  * ![b](https://img.shields.io/badge/*-b-lightgrey.svg) Coordinate/Compass Search (**[CS](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ds/cs.py)**): [[Torczon, 1997, SIAM-JO](https://epubs.siam.org/doi/abs/10.1137/S1052623493250780); [Fermi&Metropolis, 1952](https://www.osti.gov/servlets/purl/4377177)]
* **Random (stochastic) Search ([RS](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/rs.py))** [ e.g., [Murphy, 2023](https://probml.github.io/pml-book/book2.html); [Gao&Sener, 2022, ICML](https://proceedings.mlr.press/v162/gao22f.html); [Russell&Norvig, 2021](http://aima.cs.berkeley.edu/); [Nesterov&Spokoiny, 2017, FoCM](https://link.springer.com/article/10.1007/s10208-015-9296-2); [Bergstra&Bengio, 2012, JMLR](https://www.jmlr.org/papers/v13/bergstra12a.html); [Schmidhuber et al., 2001](https://ml.jku.at/publications/older/ch9.pdf); [Cvijović&Klinowski, 1995, Science](https://www.science.org/doi/abs/10.1126/science.267.5198.664); [Rastrigin, 1986](https://link.springer.com/content/pdf/10.1007/BFb0007129.pdf); [Solis&Wets, 1981, MOOR](https://pubsonline.informs.org/doi/abs/10.1287/moor.6.1.19); [Brooks, 1958, OR](https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244); [Ashby, 1952](https://link.springer.com/book/10.1007/978-94-015-1320-3) ]
  * ![lso](https://img.shields.io/badge/***-lso-orange.svg)  BErnoulli Smoothing (**[BES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/bes.py)**) [[Gao&Sener, 2022, ICML](https://proceedings.mlr.press/v162/gao22f.html)]
  * ![lso](https://img.shields.io/badge/***-lso-orange.svg) Gaussian Smoothing (**[GS](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/gs.py)**) [[Nesterov&Spokoiny, 2017, FoCM](https://link.springer.com/article/10.1007/s10208-015-9296-2)]
  * ![c](https://img.shields.io/badge/**-c-blue.svg) Simple Random Search (**[SRS](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/srs.py)**) [[Rosenstein&Barto, 2001, IJCAI](https://dl.acm.org/doi/abs/10.5555/1642194.1642206)]
  * ![b](https://img.shields.io/badge/*-b-lightgrey.svg) Annealed Random Hill Climber (**[ARHC](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/arhc.py)**) [e.g. [Russell&Norvig, 2021](http://aima.cs.berkeley.edu/); [Schaul et al., 2010, JMLR](https://jmlr.org/papers/v11/schaul10a.html)]
  * ![b](https://img.shields.io/badge/*-b-lightgrey.svg) Random Hill Climber (**[RHC](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/rhc.py)**) [e.g. [Russell&Norvig, 2021](http://aima.cs.berkeley.edu/); [Schaul et al., 2010, JMLR](https://jmlr.org/papers/v11/schaul10a.html)]
  * ![b](https://img.shields.io/badge/*-b-lightgrey.svg) Pure Random Search (**[PRS](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/prs.py)**) [e.g. [Bergstra&Bengio, 2012, JMLR](https://www.jmlr.org/papers/v13/bergstra12a.html); [Schmidhuber et al., 2001](https://ml.jku.at/publications/older/ch9.pdf); [Brooks, 1958, OR](https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244); [Ashby, 1952](https://link.springer.com/book/10.1007/978-94-015-1320-3)]

### Various Variants and Versions

Here various *variants* and *versions* of these above algorithm families of BBO
are given below and still classified into three groups (*lso*, *c*, and *b*).

#### Evolution Strategies ([ES](https://tinyurl.com/ycd8dyz4))

* ![lso](https://img.shields.io/badge/***-lso-orange.svg) LMCMA
  [[Loshchilov, 2017, ECJ](https://direct.mit.edu/evco/article-abstract/25/1/143/1041/LM-CMA-An-Alternative-to-L-BFGS-for-Large-Scale)]
* ![lso](https://img.shields.io/badge/***-lso-orange.svg) LMCMAES
  [[Loshchilov, 2014, GECCO](https://dl.acm.org/doi/abs/10.1145/2576768.2598294)]
* ![lso](https://img.shields.io/badge/***-lso-orange.svg) R1ES
  [[Li&Zhang, 2018, TEVC](https://ieeexplore.ieee.org/document/8080257),
  [Li&Zhang, 2016, PPSN](https://link.springer.com/chapter/10.1007/978-3-319-45823-6_70)]
* ![lso](https://img.shields.io/badge/***-lso-orange.svg) OPOA2015
  [[Krause&Igel, 2015, FOGA](https://dl.acm.org/doi/abs/10.1145/2725494.2725496)]
* ![lso](https://img.shields.io/badge/***-lso-orange.svg) OPOA2010
  [[Arnold&Hansen, 2010, GECCO](https://dl.acm.org/doi/abs/10.1145/1830483.1830556),
  [Jastrebski&Arnold, 2006, CEC](https://ieeexplore.ieee.org/abstract/document/1688662)]
* ![lso](https://img.shields.io/badge/***-lso-orange.svg) CCMAES2009
  [[Suttorp et al., 2009, MLJ](https://link.springer.com/article/10.1007/s10994-009-5102-1)]
* ![lso](https://img.shields.io/badge/***-lso-orange.svg) OPOC2009
  [[Suttorp et al., 2009, MLJ](https://link.springer.com/article/10.1007/s10994-009-5102-1)]
* ![lso](https://img.shields.io/badge/***-lso-orange.svg) OPOC2006
  [[Igel et al., 2006, GECCO](https://dl.acm.org/doi/abs/10.1145/1143997.1144082)]
* ![c](https://img.shields.io/badge/**-c-blue.svg) MAES
  [[Beyer&Sendhoff, 2017, TEVC](https://ieeexplore.ieee.org/abstract/document/7875115/)]
* ![c](https://img.shields.io/badge/**-c-blue.svg) DDCMA
  (Diagonal Decoding CMA)
  [[Akimoto&Hansen, 2020,
  ECJ](https://direct.mit.edu/evco/article/28/3/405/94999/Diagonal-Acceleration-for-Covariance-Matrix)]
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) SAMAES
  (Self-Adaptation MAES)
  [[Beyer, 2020, GECCO](https://dl.acm.org/doi/abs/10.1145/3377929.3389870)]
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) SAES
  (Self-Adaptation ES)
  [[Beyer, 2020, GECCO](https://dl.acm.org/doi/abs/10.1145/3377929.3389870),
  [Beyer, 2007, Scholarpedia](http://www.scholarpedia.org/article/Evolution_strategies)]
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) CSAES
  (Cumulative Step-size Adaptation ES)
  [[Hansen et al., 2015, Springer Handbook of
  Computational Intelligence](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44),
  [Ostermeier et al., 1994, PPSN](https://link.springer.com/chapter/10.1007/3-540-58484-6_263)]
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) DSAES
  (Derandomized SAES)
  [[Hansen et al., 2015, Springer Handbook of
  Computational Intelligence](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44),
  [Ostermeier et al., 1994,
  ECJ](https://direct.mit.edu/evco/article-abstract/2/4/369/1407/A-Derandomized-Approach-to-Self-Adaptation-of)]
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) SSAES
  (Schwefel's SAES) [[Hansen et al., 2015, Springer Handbook of
  Computational Intelligence](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44),
  [Beyer&Schwefel, 2002, NACO](https://link.springer.com/article/10.1023/A:1015059928466),
  [Schwefel, 1988, Research Reports in Physics](https://link.springer.com/chapter/10.1007/978-3-642-73953-8_8),
  [Schwefel, 1984, AOR](https://link.springer.com/article/10.1007/BF01876146)]
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) RES
  (Rechenberg's (1+1)-ES with 1/5 success rule) [e.g.,
  [Hansen et al., 2015, Springer Handbook of
  Computational Intelligence](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44),
  [Kern et al., 2004, NACO](https://link.springer.com/article/10.1023/B:NACO.0000023416.59689.4e),
  [Rechenberg, 1989, Lecture Notes in Engineering](https://link.springer.com/chapter/10.1007/978-3-642-83814-9_6),
  [Rechenberg, 1984, Springer Series in Synergetics](https://link.springer.com/chapter/10.1007/978-3-642-69540-7_13),
  [Schumer&Steiglitz, 1968, TAC](https://ieeexplore.ieee.org/abstract/document/1098903)]

#### Natural ES (NES)

* ![lso](https://img.shields.io/badge/***-lso-orange.svg) SNES
  (Separable NES)
  [[Schaul et al., 2011, GECCO](https://dl.acm.org/doi/abs/10.1145/2001576.2001692)]
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) ONES
  (Original NES)
  [[Wierstra et al., 2008, CEC](https://ieeexplore.ieee.org/abstract/document/4631255)]
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) SGES (Search Gradient ES)
  [[Wierstra et al., 2008, CEC](https://ieeexplore.ieee.org/abstract/document/4631255)]

#### Estimation of Distribution Algorithms (EDA)

* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) AEMNA
  (Adaptive Estimation of Multivariate Normal Algorithm)
  [[Larrañaga&Lozano, 2002](https://link.springer.com/book/10.1007/978-1-4615-1539-5)]
* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) EMNA
  [[Larrañaga&Lozano, 2002](https://link.springer.com/book/10.1007/978-1-4615-1539-5)]

#### Cross-Entropy Method (CEM)

* ![b](https://img.shields.io/badge/*-b-lightgrey.svg) MRAS
  (Model Reference Adaptive Search)
  [[Hu et al., 2007, OR](https://pubsonline.informs.org/doi/abs/10.1287/opre.1060.0367)]

#### An Open Interface to New and Missed BBO

For any new/missed BBO, we have provided **a unified API** to freely add if they can well
satisfy the [design philosophy](https://pypop.readthedocs.io/en/latest/design-philosophy.html)
*widely* recognized in the scientific research community. Note that currently both Ant Colony
Optimization ([ACO](https://www.sciencedirect.com/science/article/pii/B9781558603776500396))
and Tabu Search ([TS](https://www.science.org/doi/10.1126/science.267.5198.664)) are not
covered here, since they work mainly in *[discrete or combinatorial](https://tinyurl.com/327auv56)*
search spaces in many cases. Furthermore, both brute-force (exhaustive) search and grid search
are also excluded here, since it works only for *very low* (typically < 10) dimensions. In some
near-future version, we may consider to add other BBO (such as the well-known [Simultaneous
Perturbation Stochastic Approximation (SPSA)](https://www.jhuapl.edu/SPSA/) algorithm) into this
open-source library. Please refer to the [online development
guide](https://pypop.readthedocs.io/en/latest/development-guide.html) for more details.

## Computational Efficiency

For large-scale optimization (LSO), computational efficiency is an indispensable performance criterion of BBO/DFO/ZOO [in the post-Moore
era](https://www.science.org/doi/10.1126/science.aam9744). To obtain high-performance computation as much as possible,
[NumPy](https://www.nature.com/articles/s41586-020-2649-2) is heavily used in this library as the base of numerical computation along
with [SciPy](https://www.nature.com/articles/s41592-019-0686-2) and [scikit-learn](https://scikit-learn.org). Sometimes
[Numba](https://numba.pydata.org/) is also utilized, in order to further accelerate the *wall-clock* time.

## Folder Structure

The first-level folder structure of this library **PyPop7** is presented below:

* `.circleci`: for automatic testing based on [pytest](https://docs.pytest.org/en/8.2.x/).
  * `config.yml`: configuration file in [CircleCI](https://circleci.com/).
* `.github`: all configuration files for GitHub.
  * `workflows`: for [https://github.com/Evolutionary-Intelligence/pypop/actions](https://github.com/Evolutionary-Intelligence/pypop/actions).
* `docs`: for [online](https://pypop.readthedocs.io/) documentations.
* `pypop7`: all [Python](https://www.python.org/) source code of BBO. 
* `tutorials`: a set of tutorials.
* `.gitignore`: for [GitHub](https://github.com/).
* `.readthedocs.yaml`: for [readthedocs](https://docs.readthedocs.io/en/stable/).
* `CODE_OF_CONDUCT.md`: code of conduct.
* `LICENSE`: open-source license.
* `README.md`: basic information of this library.
* `pyproject.toml`: for [PyPI](https://pypi.org/) (used by `setup.cfg` as `build-system`).
* `requirements.txt`: only for [development]().
* `setup.cfg`: only for [PyPI](https://pypi.org/) (used via `pyproject.toml`).

## References

For each **population-based** algorithm family, we are providing several *representative*
applications published on some (rather all) [top-tier](https://github.com/Evolutionary-Intelligence/DistributedEvolutionaryComputation)
journals/conferences (such as, [Nature](https://www.nature.com/),
[Science](https://www.science.org/journal/science),
[PNAS](https://www.pnas.org/),
[PRL](https://journals.aps.org/prl/),
[JACS](https://pubs.acs.org/journal/jacsat),
[JACM](https://dl.acm.org/journal/jacm),
[PIEEE](https://proceedingsoftheieee.ieee.org/),
[JMLR](https://www.jmlr.org/),
[ICML](https://icml.cc/),
[NeurIPS](https://neurips.cc/),
[ICLR](https://iclr.cc/),
[CVPR](https://www.thecvf.com/),
[ICCV](https://www.thecvf.com/),
[RSS](https://www.roboticsproceedings.org/index.html),
just to name a few),
reported in the (actively-updated) paper list called
[DistributedEvolutionaryComputation](https://github.com/Evolutionary-Intelligence/DistributedEvolutionaryComputation).

* Derivative-Free Optimization (DFO) / Zeroth-Order Optimization (ZOO)
  * Berahas, A.S., et al., 2022.
    [A theoretical and empirical comparison of gradient approximations in derivative-free optimization](https://link.springer.com/article/10.1007/s10208-021-09513-z).
    FoCM, 22(2), pp.507-560.
  * Larson, J., et al., 2019.
    [Derivative-free optimization methods](https://www.cambridge.org/core/journals/acta-numerica/article/abs/derivativefree-optimization-methods/84479E2B03A9BFFE0F9CD46CF9FCD289).
    Acta Numerica, 28, pp.287-404.
  * Nesterov, Y., 2018.
    [Lectures on convex optimization](https://tinyurl.com/4yccr5k8).
    Cham, Switzerland: Springer.
    "**most of the achievements in Structural Optimization are firmly supported
    by the fundamental methods of Black-Box Convex Optimization.**"
  * Nesterov, Y. and Spokoiny, V., 2017.
    [Random gradient-free minimization of convex functions](https://link.springer.com/article/10.1007/s10208-015-9296-2).
    FoCM, 17(2), pp.527-566.
  * Rios, L.M. and Sahinidis, N.V., 2013.
    [Derivative-free optimization: A review of algorithms and comparison of software implementations](https://link.springer.com/article/10.1007/s10898-012-9951-y).
    JGO, 56, pp.1247-1293.
  * Conn, A.R., et al., 2009.
    [Introduction to derivative-free optimization](https://epubs.siam.org/doi/book/10.1137/1.9780898718768).
    SIAM.
  * Kolda, T.G., et al., 2003.
    [Optimization by direct search: New perspectives on some classical and modern methods](https://epubs.siam.org/doi/abs/10.1137/S003614450242889).
    SIAM Review, 45(3), pp.385-482.
* Evolutionary Computation (EC) and Swarm Intelligence (SI)
  * Eiben, A.E. and Smith, J., 2015. [From evolutionary computation to the evolution of things](https://www.nature.com/articles/nature14544.). Nature, 521(7553), pp.476-482. [ [http://www.evolutionarycomputation.org/](http://www.evolutionarycomputation.org/) ]
  * Miikkulainen, R. and Forrest, S., 2021. [A biological perspective on evolutionary computation](https://www.nature.com/articles/s42256-020-00278-8). Nature Machine Intelligence, 3(1), pp.9-15.
  * Hansen, N. and Auger, A., 2014. [Principled design of continuous stochastic search: From theory to practice](https://link.springer.com/chapter/10.1007/978-3-642-33206-7_8). Theory and Principled Methods for the Design of Metaheuristics, pp.145-180.
  * De Jong, K.A., 2006. [Evolutionary computation: A unified approach](https://mitpress.mit.edu/9780262041942/evolutionary-computation/). MIT Press.
  * Beyer, H.G. and Deb, K., 2001. [On self-adaptive features in real-parameter evolutionary algorithms](https://ieeexplore.ieee.org/abstract/document/930314). TEVC, 5(3), pp.250-270.
  * Salomon, R., 1998. [Evolutionary algorithms and gradient search: Similarities and differences](https://ieeexplore.ieee.org/abstract/document/728207). TEVC, 2(2), pp.45-55.
  * Fogel, D.B., 1998. [Evolutionary computation: The fossil record](https://ieeexplore.ieee.org/book/5263042). IEEE Press.
  * Back, T., Fogel, D.B. and Michalewicz, Z. eds., 1997. [Handbook of Evolutionary Computation](https://doi.org/10.1201/9780367802486). CRC Press.
  * Wolpert, D.H. and Macready, W.G., 1997. [No free lunch theorems for optimization](https://ieeexplore.ieee.org/document/585893). TEVC, 1(1), pp.67-82.
  * Bäck, T. and Schwefel, H.P., 1993. [An overview of evolutionary algorithms for parameter optimization](https://direct.mit.edu/evco/article-abstract/1/1/1/1092/An-Overview-of-Evolutionary-Algorithms-for). ECJ, 1(1), pp.1-23.
  * Forrest, S., 1993. [Genetic algorithms: Principles of natural selection applied to computation](https://www.science.org/doi/10.1126/science.8346439). Science, 261(5123), pp.872-878.
  * [Taxonomy](https://link.springer.com/article/10.1007/s11047-020-09820-4)
* Benchmarking [ [benchmarking-network](https://sites.google.com/view/benchmarking-network) + [iohprofiler](https://iohprofiler.github.io/) ]
  * Andrés-Thió, N., Audet, C., et al., 2024. Solar: [A solar thermal power plant simulator for blackbox optimization benchmarking](https://arxiv.org/abs/2406.00140). arXiv preprint arXiv:2406.00140.
  * Kudela, J., 2022. [A critical problem in benchmarking and analysis of evolutionary computation methods](https://www.nature.com/articles/s42256-022-00579-0). Nature Machine Intelligence, 4(12), pp.1238-1245.
  * Meunier, L., Rakotoarison, H., Wong, P.K., Roziere, B., Rapin, J., Teytaud, O., Moreau, A. and Doerr, C., 2022. [Black-box optimization revisited: Improving algorithm selection wizards through massive benchmarking](https://ieeexplore.ieee.org/abstract/document/9524335). TEVC, 26(3), pp.490-500.
  * Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T. and Brockhoff, D., 2021. [COCO: A platform for comparing continuous optimizers in a black-box setting](https://www.tandfonline.com/doi/full/10.1080/10556788.2020.1808977). Optimization Methods and Software, 36(1), pp.114-144.
  * Auger, A. and Hansen, N., 2021, July. [Benchmarking: State-of-the-art and beyond](https://dl.acm.org/doi/abs/10.1145/3449726.3461424). GECCO Companion (pp. 339-340). ACM.
  * Varelas, K., El Hara, O.A., Brockhoff, D., Hansen, N., Nguyen, D.M., Tušar, T. and Auger, A., 2020. [Benchmarking large-scale continuous optimizers: The bbob-largescale testbed, a COCO software guide and beyond](https://www.sciencedirect.com/science/article/abs/pii/S156849462030675X). ASC, 97, p.106737.
  * Hansen, N., Ros, R., Mauny, N., Schoenauer, M. and Auger, A., 2011. [Impacts of invariance in search: When CMA-ES and PSO face ill-conditioned and non-separable problems](https://www.sciencedirect.com/science/article/pii/S1568494611000974). ASC, 11(8), pp.5755-5769.
  * Moré, J.J. and Wild, S.M., 2009. [Benchmarking derivative-free optimization algorithms](https://epubs.siam.org/doi/abs/10.1137/080724083). SIOPT, 20(1), pp.172-191.
  * Whitley, D., Rana, S., Dzubera, J. and Mathias, K.E., 1996. [Evaluating evolutionary algorithms](https://www.sciencedirect.com/science/article/pii/0004370295001247). AIJ, 85(1-2), pp.245-276.
  * Salomon, R., 1996. [Re-evaluating genetic algorithm performance under coordinate rotation of benchmark functions. A survey of some theoretical and practical aspects of genetic algorithms](https://www.sciencedirect.com/science/article/abs/pii/0303264796016218). BioSystems, 39(3), pp.263-278.
  * Fogel, D.B. and Beyer, H.G., 1995. [A note on the empirical evaluation of intermediate recombination](https://direct.mit.edu/evco/article-abstract/3/4/491/749/A-Note-on-the-Empirical-Evaluation-of-Intermediate). ECJ, 3(4), pp.491-495.
  * Moré, J.J., Garbow, B.S. and Hillstrom, K.E., 1981. [Testing unconstrained optimization software](https://dl.acm.org/doi/10.1145/355934.355936). TOMS, 7(1), pp.17-41.
* Evolution Strategy (ES) [ [A visual guide to ES](https://blog.otoro.net/2017/10/29/visual-evolution-strategies/) + [[Li et al., 2020]](https://www.sciencedirect.com/science/article/abs/pii/S221065021930584X) + [[Akimoto&Hansen, 2022, GECCO-Companion]](http://www.cmap.polytechnique.fr/~nikolaus.hansen/gecco-2022-cma-tutorial.pdf) ]
  * Akimoto, Y., Auger, A., Glasmachers, T. and Morinaga, D., 2022. [Global linear convergence of evolution strategies on more than smooth strongly convex functions](https://epubs.siam.org/doi/abs/10.1137/20M1373815). SIOPT, 32(2), pp.1402-1429.
  * Glasmachers, T. and Krause, O., 2022. [Convergence analysis of the Hessian estimation evolution strategy](https://direct.mit.edu/evco/article-abstract/30/1/27/102711/Convergence-Analysis-of-the-Hessian-Estimation). ECJ, 30(1), pp.27-50.
  * He, X., Zheng, Z. and Zhou, Y., 2021. [MMES: Mixture model-based evolution strategy for large-scale optimization](https://ieeexplore.ieee.org/abstract/document/9244595). TEVC, 25(2), pp.320-333.
  * Akimoto, Y. and Hansen, N., 2020. [Diagonal acceleration for covariance matrix adaptation evolution strategies](https://direct.mit.edu/evco/article/28/3/405/94999/Diagonal-Acceleration-for-Covariance-Matrix). ECJ, 28(3), pp.405-435.
  * Beyer, H.G., 2020, July. [Design principles for matrix adaptation evolution strategies](https://dl.acm.org/doi/abs/10.1145/3377929.3389870). GECCO Companion (pp. 682-700). ACM.
  * Choromanski, K., Pacchiano, A., Parker-Holder, J. and Tang, Y., 2019. [From complexity to simplicity: Adaptive es-active subspaces for blackbox optimization](https://papers.nips.cc/paper/2019/hash/88bade49e98db8790df275fcebb37a13-Abstract.html). In NeurIPS.
  * Varelas, K., Auger, A., Brockhoff, D., Hansen, N., ElHara, O.A., Semet, Y., Kassab, R. and Barbaresco, F., 2018, September. [A comparative study of large-scale variants of CMA-ES](https://link.springer.com/chapter/10.1007/978-3-319-99253-2_1). In PPSN (pp. 3-15). Springer, Cham.
  * Li, Z. and Zhang, Q., 2018. [A simple yet efficient evolution strategy for large-scale black-box optimization](https://ieeexplore.ieee.org/abstract/document/8080257). TEVC, 22(5), pp.637-646.
  * Lehman, J., Chen, J., Clune, J. and Stanley, K.O., 2018, July. [ES is more than just a traditional finite-difference approximator](https://dl.acm.org/doi/abs/10.1145/3205455.3205474). GECCO (pp. 450-457). ACM.
  * Ollivier, Y., Arnold, L., Auger, A. and Hansen, N., 2017. [Information-geometric optimization algorithms: A unifying picture via invariance principles](https://www.jmlr.org/papers/v18/14-467.html). JMLR, 18(18), pp.1-65.
  * Loshchilov, I., 2017. [LM-CMA: An alternative to L-BFGS for large-scale black box optimization](https://direct.mit.edu/evco/article-abstract/25/1/143/1041/LM-CMA-An-Alternative-to-L-BFGS-for-Large-Scale). ECJ, 25(1), pp.143-171. [ Loshchilov, I., 2014, July. [A computationally efficient limited memory CMA-ES for large scale optimization](https://dl.acm.org/doi/abs/10.1145/2576768.2598294). GECCO (pp. 397-404). ACM. ] + [ Loshchilov, I., Glasmachers, T. and Beyer, H.G., 2019. [Large scale black-box optimization by limited-memory matrix adaptation](https://ieeexplore.ieee.org/abstract/document/8410043). TEVC, 23(2), pp.353-358. ]
  * Beyer, H.G. and Sendhoff, B., 2017. [Simplify your covariance matrix adaptation evolution strategy](https://ieeexplore.ieee.org/document/7875115). TEVC, 21(5), pp.746-759.
  * Krause, O., Arbonès, D.R. and Igel, C., 2016. [CMA-ES with optimal covariance update and storage complexity](https://proceedings.neurips.cc/paper/2016/hash/289dff07669d7a23de0ef88d2f7129e7-Abstract.html). In NeurIPS, 29, pp.370-378.
  * Akimoto, Y. and Hansen, N., 2016, July. [Projection-based restricted covariance matrix adaptation for high dimension](https://dl.acm.org/doi/abs/10.1145/2908812.2908863). GECCO (pp. 197-204). ACM.
  * Auger, A. and Hansen, N., 2016. [Linear convergence of comparison-based step-size adaptive randomized search via stability of Markov chains](https://epubs.siam.org/doi/10.1137/140984038). SIOPT, 26(3), pp.1589-1624.
  * Hansen, N., Arnold, D.V. and Auger, A., 2015. [Evolution strategies](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44). In Springer Handbook of Computational Intelligence (pp. 871-898). Springer, Berlin, Heidelberg.
  * Diouane, Y., Gratton, S. and Vicente, L.N., 2015. [Globally convergent evolution strategies](https://link.springer.com/article/10.1007/s10107-014-0793-x). MP, 152(1), pp.467-490.
  * Akimoto, Y., Auger, A. and Hansen, N., 2014, July. [Comparison-based natural gradient optimization in high dimension](https://dl.acm.org/doi/abs/10.1145/2576768.2598258). GECCO (pp. 373-380). ACM.
  * Bäck, T., Foussette, C. and Krause, P., 2013. [Contemporary evolution strategies](https://link.springer.com/book/10.1007/978-3-642-40137-4). Berlin: Springer. [ Bäck, T., 2014, July. Introduction to evolution strategies. GECCO Companion (pp. 251-280). ] + [ Wang, H., Emmerich, M. and Bäck, T., 2014, March. [Mirrored orthogonal sampling with pairwise selection in evolution strategies](https://dl.acm.org/doi/10.1145/2554850.2555089). In Proceedings of Annual ACM Symposium on Applied Computing (pp. 154-156). ]
  * Rudolph, G., 2012. [Evolutionary strategies](https://link.springer.com/referenceworkentry/10.1007/978-3-540-92910-9_22). In Handbook of Natural Computing (pp. 673-698). Springer Berlin, Heidelberg.
  * Akimoto, Y., Nagata, Y., Ono, I. and Kobayashi, S., 2012. [Theoretical foundation for CMA-ES from information geometry perspective](https://link.springer.com/article/10.1007/s00453-011-9564-8). Algorithmica, 64(4), pp.698-716. [ Akimoto, Y., Nagata, Y., Ono, I. and Kobayashi, S., 2010, September. [Bidirectional relation between CMA evolution strategies and natural evolution strategies](https://link.springer.com/chapter/10.1007/978-3-642-15844-5_16). In PPSN (pp. 154-163). Springer, Berlin, Heidelberg. ] + [ Akimoto, Y., 2011. [Design of evolutionary computation for continuous optimization](https://drive.google.com/file/d/18PW9syYDy-ndJA7wBmE2hRlxXJRBTTir/view). Doctoral Dissertation, Tokyo Institute of Technology. ]
  * Arnold, D.V. and Hansen, N., 2010, July. [Active covariance matrix adaptation for the (1+1)-CMA-ES](https://dl.acm.org/doi/abs/10.1145/1830483.1830556). GECCO (pp. 385-392). ACM.
  * Arnold, D.V. and MacLeod, A., 2006, July. [Hierarchically organised evolution strategies on the parabolic ridge](https://dl.acm.org/doi/abs/10.1145/1143997.1144080). GECCO (pp. 437-444). ACM.
  * Igel, C., Suttorp, T. and Hansen, N., 2006, July. [A computational efficient covariance matrix update and a (1+1)-CMA for evolution strategies](https://dl.acm.org/doi/abs/10.1145/1143997.1144082). GECCO (pp. 453-460). ACM. [ Suttorp, T., Hansen, N. and Igel, C., 2009. [Efficient covariance matrix update for variable metric evolution strategies](https://link.springer.com/article/10.1007/s10994-009-5102-1). MLJ, 75(2), pp.167-197. ] + [ Krause, O. and Igel, C., 2015, January. [A more efficient rank-one covariance matrix update for evolution strategies](https://dl.acm.org/doi/abs/10.1145/2725494.2725496). In FOGA (pp. 129-136). ACM. ]
  * Beyer, H.G. and Schwefel, H.P., 2002. [Evolution strategies–A comprehensive introduction](https://link.springer.com/article/10.1023/A:1015059928466). Natural Computing, 1(1), pp.3-52.
  * Hansen, N. and Ostermeier, A., 1996, May. [Adapting arbitrary normal mutation distributions in evolution strategies: The covariance matrix adaptation](https://ieeexplore.ieee.org/abstract/document/542381). In CEC (pp. 312-317). IEEE. [ Hansen, N. and Ostermeier, A., 2001. [Completely derandomized self-adaptation in evolution strategies](https://direct.mit.edu/evco/article-abstract/9/2/159/892/Completely-Derandomized-Self-Adaptation-in). ECJ, 9(2), pp.159-195. ] + [ Hansen, N., Müller, S.D. and Koumoutsakos, P., 2003. [Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES)](https://direct.mit.edu/evco/article-abstract/11/1/1/1139/Reducing-the-Time-Complexity-of-the-Derandomized). ECJ, 11(1), pp.1-18. ] + [ Auger, A. and Hansen, N., 2005, September. [A restart CMA evolution strategy with increasing population size](https://ieeexplore.ieee.org/abstract/document/1554902). In CEC (pp. 1769-1776). IEEE. ] + [ Hansen, N. and Auger, A., 2014. [Principled design of continuous stochastic search: From theory to practice](https://link.springer.com/chapter/10.1007/978-3-642-33206-7_8). In Theory and Principled Methods for the Design of Metaheuristics (pp. 145-180). Springer, Berlin, Heidelberg. ]
  * Rudolph, G., 1992. [On correlated mutations in evolution strategies](https://ls11-www.cs.tu-dortmund.de/people/rudolph/publications/papers/PPSN92.pdf). In PPSN (pp. 105-114).
  * Schwefel, H.P., 1984. [Evolution strategies: A family of non-linear optimization techniques based on imitating some principles of organic evolution](https://link.springer.com/article/10.1007/BF01876146). Annals of Operations Research, 1(2), pp.165-167. [ Schwefel, H.P., 1988. [Collective intelligence in evolving systems](https://link.springer.com/chapter/10.1007/978-3-642-73953-8_8). In Ecodynamics (pp. 95-100). Springer, Berlin, Heidelberg. ]
  * Rechenberg, I., 1984. [The evolution strategy. A mathematical model of darwinian evolution](https://link.springer.com/chapter/10.1007/978-3-642-69540-7_13). In Synergetics—from Microscopic to Macroscopic Order (pp. 122-132). Springer, Berlin, Heidelberg. [ Rechenberg, I., 1989. [Evolution strategy: Nature’s way of optimization](https://link.springer.com/chapter/10.1007/978-3-642-83814-9_6). In Optimization: Methods and Applications, Possibilities and Limitations (pp. 106-126). Springer, Berlin, Heidelberg. ]
  * Applications: e.g., [Deng et al., 2023](https://www.chimechallenge.org/current/workshop/papers/CHiME_2023_DASR_deng.pdf); [Zhang et al., 2023, NeurIPS](https://arxiv.org/pdf/2310.18622.pdf); [Tjanaka et al., 2023, IEEE-LRA](https://scalingcmamae.github.io/); [Yu et al., 2023, IJCAI](https://www.ijcai.org/proceedings/2023/0187.pdf); [Zhu et al., 2023, IEEE/ASME-TMECH](https://ieeexplore.ieee.org/abstract/document/10250896); [Fadini et al., 2023](https://laas.hal.science/hal-04162737/file/versatileQuadrupedCodesign_preprint.pdf); [Ma et al., 2023](https://cs.brown.edu/~gdk/pubs/skillgen_verbs.pdf); [Kim et al., 2023, Science Robotics](https://www.science.org/doi/10.1126/scirobotics.add1053); [Slade et al., 2022, Nature](https://www.nature.com/articles/s41586-022-05191-1); [Sun et al., 2022, ICML](https://proceedings.mlr.press/v162/sun22e.html); [Tjanaka et al., 2022, GECCO](https://dl.acm.org/doi/10.1145/3512290.3528705); [Wang&Ponce, 2022, GECCO](https://dl.acm.org/doi/10.1145/3512290.3528725), [Tian&Ha, 2022, EvoStar](https://link.springer.com/chapter/10.1007/978-3-031-03789-4_18); [Hansel et al., 2021](https://link.springer.com/chapter/10.1007/978-3-030-41188-6_7); [Anand et al., 2021, MLST](https://iopscience.iop.org/article/10.1088/2632-2153/abf3ac); [Nomura et al., 2021, AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17109); [Zheng et al., 2021, IEEE-ASRU](https://ieeexplore.ieee.org/abstract/document/9688232); [Liu et al., 2019, AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/4345); [Dong et al., 2019, CVPR](https://ieeexplore.ieee.org/document/8953400); [Ha&Schmidhuber, 2018, NeurIPS](https://papers.nips.cc/paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html); [Müller&Glasmachers, 2018, PPSN](https://link.springer.com/chapter/10.1007/978-3-319-99259-4_33); [Chrabąszcz et al., 2018, IJCAI](https://www.ijcai.org/proceedings/2018/197); [OpenAI, 2017](https://openai.com/blog/evolution-strategies/); [Zhang et al., 2017, Science](https://www.science.org/doi/10.1126/science.aal5054).
* **Natural Evolution Strategies (NES)**
  * Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014. [Natural evolution strategies](https://jmlr.org/papers/v15/wierstra14a.html). JMLR, 15(1), pp.949-980.
  * Beyer, H.G., 2014.
    Convergence analysis of evolutionary algorithms that are based on the paradigm of information geometry.
    ECJ, 22(4), pp.679-709.
  * Schaul, T., 2011. [Studies in continuous black-box optimization](https://people.idsia.ch/~schaul/publications/thesis.pdf). Doctoral Dissertation, Technische Universität München.
  * Yi, S., Wierstra, D., Schaul, T. and Schmidhuber, J., 2009, June. [Stochastic search using the natural gradient](https://dl.acm.org/doi/10.1145/1553374.1553522). ICML (pp. 1161-1168).
  * Wierstra, D., Schaul, T., Peters, J. and Schmidhuber, J., 2008, June. [Natural evolution strategies](https://ieeexplore.ieee.org/abstract/document/4631255). CEC (pp. 3381-3387). IEEE.
  * Applications: e.g., [Yu et al., USENIX Security](https://www.usenix.org/conference/usenixsecurity23/presentation/yuzhiyuan); [Flageat et al., 2023](https://arxiv.org/abs/2303.06137); [Yan et al., 2023](https://arxiv.org/abs/2302.04477); [Feng et al., 2023](https://arxiv.org/abs/2303.06280); [Wei et al., 2022, IJCV](https://link.springer.com/article/10.1007/s11263-022-01604-w); [Agarwal et al., 2022, ICRA](https://ieeexplore.ieee.org/abstract/document/9811565); [Farid et al., 2022, CoRL](https://proceedings.mlr.press/v164/farid22a.html); [Feng et al., 2022, CVPR](https://openaccess.thecvf.com/content/CVPR2022/html/Feng_Boosting_Black-Box_Attack_With_Partially_Transferred_Conditional_Adversarial_Distribution_CVPR_2022_paper.html); [Berliner et al., 2022, ICLR](https://openreview.net/forum?id=JJCjv4dAbyL); [Kirsch et al., 2022, AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/20681); [Jain et al., 2022, USENIX Security](https://www.usenix.org/conference/usenixsecurity22/presentation/jain); [Ilyas et al., 2018, ICML](https://proceedings.mlr.press/v80/ilyas18a.html).
* Estimation of Distribution Algorithm (EDA) [ [MIMIC [NeurIPS-1996]](https://proceedings.neurips.cc/paper/1996/hash/4c22bd444899d3b6047a10b20a2f26db-Abstract.html) + [BOA [GECCO-1999]](https://dl.acm.org/doi/abs/10.5555/2933923.2933973) + [[ECJ-2005]](https://direct.mit.edu/evco/article-abstract/13/1/99/1198/Drift-and-Scaling-in-Estimation-of-Distribution) ]
  * Brookes, D., Busia, A., Fannjiang, C., Murphy, K. and Listgarten, J., 2020, July. [A view of estimation of distribution algorithms through the lens of expectation-maximization](https://dl.acm.org/doi/abs/10.1145/3377929.3389938). GECCO Companion (pp. 189-190). ACM.
  * Kabán, A., Bootkrajang, J. and Durrant, R.J., 2016. [Toward large-scale continuous EDA: A random matrix theory perspective](https://direct.mit.edu/evco/article-abstract/24/2/255/1016/Toward-Large-Scale-Continuous-EDA-A-Random-Matrix). ECJ, 24(2), pp.255-291.
  * Pelikan, M., Hauschild, M.W. and Lobo, F.G., 2015. [Estimation of distribution algorithms](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_45). In Springer Handbook of Computational Intelligence (pp. 899-928). Springer, Berlin, Heidelberg.
  * Dong, W., Chen, T., Tiňo, P. and Yao, X., 2013. [Scaling up estimation of distribution algorithms for continuous optimization](https://ieeexplore.ieee.org/document/6461934). TEVC, 17(6), pp.797-822.
  * Hauschild, M. and Pelikan, M., 2011. [An introduction and survey of estimation of distribution algorithms](https://www.sciencedirect.com/science/article/abs/pii/S2210650211000435). Swarm and Evolutionary Computation, 1(3), pp.111-128.
  * Teytaud, F. and Teytaud, O., 2009, July. [Why one must use reweighting in estimation of distribution algorithms](https://dl.acm.org/doi/10.1145/1569901.1569964). GECCO (pp. 453-460).
  * Larrañaga, P. and Lozano, J.A. eds., 2001. [Estimation of distribution algorithms: A new tool for evolutionary computation](https://link.springer.com/book/10.1007/978-1-4615-1539-5). Springer Science & Business Media.
  * Mühlenbein, H., 1997. [The equation for response to selection and its use for prediction](https://tinyurl.com/yt78c786).  ECJ, 5(3), pp.303-346.
  * Baluja, S. and Caruana, R., 1995. [Removing the genetics from the standard genetic algorithm](https://www.sciencedirect.com/science/article/pii/B9781558603776500141). ICML (pp. 38-46). Morgan Kaufmann.
* Cross-Entropy Method (CEM)
  * Pinneri, C., Sawant, S., Blaes, S., Achterhold, J., Stueckler, J., Rolinek, M. and Martius, G., 2021, October. [Sample-efficient cross-entropy method for real-time planning](https://proceedings.mlr.press/v155/pinneri21a.html). In Conference on Robot Learning (pp. 1049-1065). PMLR.
  * Amos, B. and Yarats, D., 2020, November. [The differentiable cross-entropy method](https://proceedings.mlr.press/v119/amos20a.html). ICML (pp. 291-302). PMLR.
  * Rubinstein, R.Y. and Kroese, D.P., 2016. [Simulation and the Monte Carlo method (Third Edition)](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118631980). John Wiley & Sons.
  * Hu, J., et al., 2007.
    [A model reference adaptive search method for global optimization](https://pubsonline.informs.org/doi/abs/10.1287/opre.1060.0367).
    OR, 55(3), pp.549-568.
  * De Boer, P.T., Kroese, D.P., Mannor, S. and Rubinstein, R.Y., 2005. [A tutorial on the cross-entropy method](https://link.springer.com/article/10.1007/s10479-005-5724-z). Annals of Operations Research, 134(1), pp.19-67.
  * Rubinstein, R.Y. and Kroese, D.P., 2004. [The cross-entropy method: A unified approach to combinatorial optimization, Monte-Carlo simulation, and machine learning](https://link.springer.com/book/10.1007/978-1-4757-4321-0). New York: Springer.
  * Mannor, S., Rubinstein, R.Y. and Gat, Y., 2003. [The cross entropy method for fast policy search](https://dl.acm.org/doi/abs/10.5555/3041838.3041903). ICML (pp. 512-519).
  * Applications: e.g., [Wang&Ba,2020, ICLR](https://openreview.net/forum?id=H1exf64KwH); [Hafner et al., 2019, ICML](https://proceedings.mlr.press/v97/hafner19a.html); [Pourchot&Sigaud, 2019, ICLR](https://openreview.net/forum?id=BkeU5j0ctQ); [Simmons-Edler et al., 2019, ICML-RL4RealLife](https://openreview.net/forum?id=SyeHbtgSiN); [Chua et al., 2018, NeurIPS](https://proceedings.neurips.cc/paper/2018/file/3de568f8597b94bda53149c7d7f5958c-Paper.pdf); [Duan et al., 2016, ICML](https://proceedings.mlr.press/v48/duan16.html); [Kobilarov, 2012, IJRR](https://journals.sagepub.com/doi/10.1177/0278364912444543).
* Differential Evolution (DE)
  * Price, K.V., 2013. [Differential evolution](https://link.springer.com/chapter/10.1007/978-3-642-30504-7_8). In Handbook of Optimization (pp. 187-214). Springer, Berlin, Heidelberg.
  * Tanabe, R. and Fukunaga, A., 2013, June. [Success-history based parameter adaptation for differential evolution](https://ieeexplore.ieee.org/document/6557555). CEC (pp. 71-78). IEEE.
  * Wang, Y., Cai, Z., and Zhang, Q. 2011. [Differential evolution with composite trial vector generation strategies and control parameters](https://doi.org/10.1109/TEVC.2010.2087271). TEVC, 15(1), pp.55–66.
  * Zhang, J., and Sanderson, A. C. 2009. [JADE: Adaptive differential evolution with optional external archive](https://ieeexplore.ieee.org/document/5208221/). TEVC, 13(5), pp.945–958.
  * Price, K.V., Storn, R.M. and Lampinen, J.A., 2005. [Differential evolution: A practical approach to global optimization](https://link.springer.com/book/10.1007/3-540-31306-0). Springer Science & Business Media.
  * Fan, H.Y. and Lampinen, J., 2003. [A trigonometric mutation operation to differential evolution](https://link.springer.com/article/10.1023/A:1024653025686). JGO, 27(1), pp.105-129.
  * Storn, R.M. and Price, K.V. 1997. [Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces](https://doi.org/10.1023/A:1008202821328). JGO, 11(4), pp.341–359.
  * Applications: e.g., [McNulty et al., 2023, PRL](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.100801); [Colombo et al., 2023, Sci. Adv.](https://www.science.org/doi/full/10.1126/sciadv.ade5839); [Lichtinger&Biggin, 2023, JCTC](https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c00140); [Liang et al., 2023, NSDI](https://www.usenix.org/system/files/nsdi23-liang-chieh-jan.pdf); [Schad et al., 2023, ApJ](https://iopscience.iop.org/article/10.3847/1538-4357/acabbd); [Hoyer et al., 2023, MNRAS](https://academic.oup.com/mnras/article/520/3/4664/6994541); [Hoyer et al., 2023, ApJL](https://iopscience.iop.org/article/10.3847/2041-8213/aca53e); [Abdelnabi&Fritz, 2023,USENIX Security](https://www.usenix.org/conference/usenixsecurity23/presentation/abdelnabi); [Kotov et al., 2023, Cell Reports](https://www.cell.com/cell-reports/pdf/S2211-1247(23)00842-2.pdf); [Sidhartha et al., 2023, CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Sidhartha_Adaptive_Annealing_for_Robust_Geometric_Estimation_CVPR_2023_paper.pdf); [Hardy et al., 2023, MNRAS](https://academic.oup.com/mnras/article-abstract/520/4/6111/6998583); [Boucher et al., 2023](https://arxiv.org/pdf/2306.07033.pdf); [Michel et al., 2023, PRA](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.042602); [Woo et al., 2023, iScience](https://www.cell.com/iscience/pdf/S2589-0042(22)02008-9.pdf); [Bozkurt et al., 2023](https://arxiv.org/pdf/2307.04830.pdf); [Ma et al., 2023, KDD](https://dl.acm.org/doi/pdf/10.1145/3580305.3599524); [Zhou et al., 2023](https://arxiv.org/pdf/2301.12738.pdf); [Czarnik et al., 2023](https://arxiv.org/pdf/2307.05302.pdf); [Katic et al., 2023, iScience](https://pdf.sciencedirectassets.com/318494/1-s2.0-S2589004222X00138/1-s2.0-S2589004222021472/main.pdf); [Khajehnejad et al., 2023, RSIF](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2022.0808); [Digman&Cornish, 2023, PRD](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.108.023022); [Rommel et al., 2023](https://arxiv.org/pdf/2308.08062.pdf); [Li et al., 2022, Science](https://www.science.org/doi/full/10.1126/science.abj2096); [Schlegelmilch et al., 2022, Psychological Review](https://psycnet.apa.org/record/2021-81557-001); [Mackin et al., 2022, Nature Communications](https://www.nature.com/articles/s41467-022-31405-1); [Liu&Wang, 2022, JSAC](https://ieeexplore.ieee.org/abstract/document/9681872); [Zhou et al., 2022, Nature Computational Science](https://www.nature.com/articles/s43588-022-00232-1); [Fischer et al., 2022, TOCHI](https://dl.acm.org/doi/full/10.1145/3524122); [Ido et al., 2022, npj Quantum Materials](https://www.nature.com/articles/s41535-022-00452-8); [Clark et al., 2022, NECO](https://direct.mit.edu/neco/article/34/7/1545/111332/Reduced-Dimension-Biophysical-Neuron-Models); [Powell et al., 2022, ApJ](https://iopscience.iop.org/article/10.3847/1538-4357/ac8934/meta); [Vo et al., 2022, ICLR](https://openreview.net/forum?id=73MEhZ0anV); [Andersson et al., 2022, ApJ](https://iopscience.iop.org/article/10.3847/1538-4357/ac64a4/meta); [Naudin et al., 2022, NECO](https://direct.mit.edu/neco/article-abstract/34/10/2075/112571/A-Simple-Model-of-Nonspiking-Neurons); [Perini et al., 2022, AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/20331); [Sterbentz et al., 2022, Physics of Fluids](https://pubs.aip.org/aip/pof/article/34/8/082109/2846942); [Mishra et al., 2021, Science](https://www.science.org/doi/full/10.1126/science.aav0780); [Tiwari et al., 2021, PRB](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.103.014432); [Mok et al., 2021, Communications Physics](https://www.nature.com/articles/s42005-021-00572-w); [Vinker et al., 2021, CVPR](https://openaccess.thecvf.com/content/ICCV2021/papers/Vinker_Unpaired_Learning_for_High_Dynamic_Range_Image_Tone_Mapping_ICCV_2021_paper.pdf); [Mehta et al., 2021, JCAP](https://iopscience.iop.org/article/10.1088/1475-7516/2021/07/033); [Trueblood et al., 2021, Psychological Review](https://psycnet.apa.org/record/2020-63299-001); [Verdonck et al., 2021, Psychological Review](https://psycnet.apa.org/record/2020-66308-001); [Robert et al., 2021, npj Quantum Information](https://www.nature.com/articles/s41534-021-00368-4); [Canton et al., 2021, ApJ](https://iopscience.iop.org/article/10.3847/1538-4357/ac2f9a/meta); [Leslie et al., 2021, PRD](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.104.123030); [Fengler et al., 2021, eLife](https://elifesciences.org/articles/65074); [Li et al., 2021, TQE](https://ieeexplore.ieee.org/abstract/document/9495278); [Chen et al., 2021, ACS Photonics](https://pubs.acs.org/doi/full/10.1021/acsphotonics.1c00915); [Menczel et al., 2021, J. Phys. A: Math. Theor.](https://iopscience.iop.org/article/10.1088/1751-8121/ac0c8f/meta); [Feng et al., 2021, JSAC](https://ieeexplore.ieee.org/abstract/document/9461747); [DES Collaboration, 2021, A&A](https://www.aanda.org/articles/aa/full_html/2021/12/aa41744-21/aa41744-21.html); [An et al., 2020, PNAS](https://www.pnas.org/doi/suppl/10.1073/pnas.1920338117).
* **Particle Swarm Optimizer (PSO) / Consensus-Based Optimization (CBO)** [ [swarm intelligence](https://www.cs.cmu.edu/~arielpro/15381f16/c_slides/781f16-26.pdf) | [scholarpedia](http://www.scholarpedia.org/article/Particle_swarm_optimization) ]
  * Fornasier, M., et al., 2024.
    [Consensus-based optimization methods converge globally](https://epubs.siam.org/doi/10.1137/22M1527805).
    SIOPT, 34(3), pp.2973-3004.
  * Bailo, R., et al., 2024.
    [CBX: Python and Julia packages for consensus-based interacting particle methods](https://joss.theoj.org/papers/10.21105/joss.06611.pdf).
    JOSS, 9(98), p.6611.
  * Sünnen, P., 2023.
    [Analysis of a consensus-based optimization method on hypersurfaces and applications](https://mediatum.ub.tum.de/doc/1647263/document.pdf).
    Doctoral Dissertation, Technische Universität München.
  * Fornasier, M., et al., 2021.
    [Consensus-based optimization on the sphere: Convergence to global minimizers and machine learning](https://jmlr.csail.mit.edu/papers/v22/21-0259.html).
    JMLR, 22(1), pp.10722-10776.
  * Carrillo, J.A., et al., 2018.
    [An analytical framework for consensus-based global optimization method](https://www.worldscientific.com/doi/abs/10.1142/S0218202518500276).
    Mathematical Models and Methods in Applied Sciences, 28(06), pp.1037-1066.
  * Pinnau, R., et al., 2017.
    [A consensus-based model for global optimization and its mean-field limit](https://www.worldscientific.com/doi/abs/10.1142/S0218202517400061).
    Mathematical Models and Methods in Applied Sciences, 27(01), pp.183-204.
  * Bonyadi, M.R. and Michalewicz, Z., 2017. [Particle swarm optimization for single objective continuous space problems: A review](https://direct.mit.edu/evco/article-abstract/25/1/1/1040/Particle-Swarm-Optimization-for-Single-Objective). ECJ, 25(1), pp.1-54.
  * Escalante, H.J., Montes, M. and Sucar, L.E., 2009. [Particle swarm model selection](https://www.jmlr.org/papers/v10/escalante09a.html). JMLR, 10(15), pp.405−440.
  * Floreano, D. and Mattiussi, C., 2008. [Bio-inspired artificial intelligence: Theories, methods, and technologies](https://mitpress.mit.edu/9780262062718/bio-inspired-artificial-intelligence/). MIT Press.
  * Poli, R., Kennedy, J. and Blackwell, T., 2007. [Particle swarm optimization](https://link.springer.com/article/10.1007/s11721-007-0002-0). Swarm Intelligence, 1(1), pp.33-57.
  * Venter, G. and Sobieszczanski-Sobieski, J., 2003. [Particle swarm optimization](https://arc.aiaa.org/doi/abs/10.2514/2.2111). AIAA Journal, 41(8), pp.1583-1589.
  * Parsopoulos, K.E. and Vrahatis, M.N., 2002. [Recent approaches to global optimization problems through particle swarm optimization](https://link.springer.com/article/10.1023/A:1016568309421). Natural Computing, 1(2), pp.235-306.
  * Clerc, M. and Kennedy, J., 2002. [The particle swarm-explosion, stability, and convergence in a multidimensional complex space](https://ieeexplore.ieee.org/abstract/document/985692). TEVC, 6(1), pp.58-73.
  * Eberhart, R.C., Shi, Y. and Kennedy, J., 2001. [Swarm intelligence](https://www.elsevier.com/books/swarm-intelligence/eberhart/978-1-55860-595-4). Elsevier.
  * Shi, Y. and Eberhart, R., 1998, May. [A modified particle swarm optimizer](https://ieeexplore.ieee.org/abstract/document/699146). CEC (pp. 69-73). IEEE.
  * Kennedy, J. and Eberhart, R., 1995, November. [Particle swarm optimization](https://ieeexplore.ieee.org/document/488968). In Proceedings of International Conference on Neural Networks (pp. 1942-1948). IEEE.
  * Eberhart, R. and Kennedy, J., 1995, October. [A new optimizer using particle swarm theory](https://ieeexplore.ieee.org/abstract/document/494215). In Proceedings of International Symposium on Micro Machine and Human Science (pp. 39-43). IEEE.
  * Interest Applications: e.g., [Grabner et al., 2023, Nature Communications](https://www.nature.com/articles/s41467-023-38943-2); [Morselli et al., 2023, IEEE-TWC](https://ieeexplore.ieee.org/abstract/document/10127621); [Reddy et al., 2023, IEEE-TC](https://ieeexplore.ieee.org/document/10005787); [Zhang et al., 2022, CVPR](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_On_Adversarial_Robustness_of_Trajectory_Prediction_for_Autonomous_Vehicles_CVPR_2022_paper.html); [Yang et al., PRL, 2022](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.128.065701); [Guan et al., 2022, PRL](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.128.186001); [Zhong et al., 2022, CVPR](https://openaccess.thecvf.com/content/CVPR2022/html/Zhong_Shadows_Can_Be_Dangerous_Stealthy_and_Effective_Physical-World_Adversarial_Attack_CVPR_2022_paper.html); [Singh&Hecke, 2021, PRL](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.248002); [Weiel, et al., 2021, Nature Mach. Intell](https://www.nature.com/articles/s42256-021-00366-3); [Wintermantel et al., 2020, PRL](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.070503); [Tang et al., 2019, TPAMI](https://ieeexplore.ieee.org/abstract/document/8386667); [Sheng et al., 2019, TPAMI](https://ieeexplore.ieee.org/abstract/document/8502935); [CMS Collaboration, 2019, JHEP](https://www.research-collection.ethz.ch/handle/20.500.11850/331761); [Wang et al., 2019, TVCG](https://ieeexplore.ieee.org/abstract/document/8826012); [Zhang et al., 2018, PRL](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.255703); [Leditzky et al., 2018, PRL](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.160501); [Pham et al., 2018, TPAMI](https://ieeexplore.ieee.org/abstract/document/8085141); [Villeneuve et al., 2017, Science](https://www.science.org/doi/10.1126/science.aam8393); [Choi et al., 2017, PRL](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.223605); [González-Echevarría, et al., 2017, TCAD](https://ieeexplore.ieee.org/document/7465733); [Zhu et al., 2017, PRL](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.106101); [Choi et al., 2017, ICCV](https://openaccess.thecvf.com/content_iccv_2017/html/Choi_Robust_Hand_Pose_ICCV_2017_paper.html); [Pickup et al., 2016, IJCV](https://link.springer.com/article/10.1007/s11263-016-0903-8); [Li et al., 2015, PRL](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.115.105502); [Sharp et al., 2015, CHI](https://dl.acm.org/doi/abs/10.1145/2702123.2702179); [Taneja et al., 2015, TPAMI](https://ieeexplore.ieee.org/abstract/document/7045528); [Zhang et al., 2015, IJCV](https://link.springer.com/article/10.1007/s11263-015-0819-8); [Meyer et al., 2015, ICCV](https://research.nvidia.com/publication/2015-12_robust-model-based-3d-head-pose-estimation); [Tompson et al., 2014, TOG](https://dl.acm.org/doi/abs/10.1145/2629500); [Baca et al., 2013, Cell](https://www.cell.com/cell/fulltext/S0092-8674(13)00343-7); [Li et al., PRL, 2013](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.136403); [Kawakami et al., 2013, IJCV](https://link.springer.com/article/10.1007/s11263-013-0632-1); [Kim et al., 2012, Nature](https://www.nature.com/articles/nature11546); [Rahmat-Samii et al., 2012, PIEEE](https://ieeexplore.ieee.org/document/6204306); [Oikonomidis et al., 2012, CVPR](https://ieeexplore.ieee.org/document/6247885); [Li et al., 2011, TPAMI](https://ieeexplore.ieee.org/abstract/document/5567109); [Zhao et al., 2011, PRL](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.215502); [Zhu et al., 2011, PRL](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.145501); [Lv et al., 2011, PRL](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.015503); [Hentschel&Sanders, 2010, PRL](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.104.063603); [Pontani&Conway, 2010, JGCD](https://arc.aiaa.org/doi/abs/10.2514/1.48475); [Zhang et al., 2008, CVPR](https://ieeexplore.ieee.org/document/4587512); [Liebelt&Schertler, 2007, CVPR](https://ieeexplore.ieee.org/abstract/document/4270192); [Hassan et al., 2005, AIAA](https://arc.aiaa.org/doi/abs/10.2514/6.2005-1897)].
* Cooperative Coevolution (CC)
  * Gomez, F., Schmidhuber, J. and Miikkulainen, R., 2008. [Accelerated neural evolution through cooperatively coevolved synapses](https://www.jmlr.org/papers/v9/gomez08a.html). JMLR, 9(31), pp.937-965.
  * Panait, L., Tuyls, K. and Luke, S., 2008. [Theoretical advantages of lenient learners: An evolutionary game theoretic perspective](https://jmlr.org/papers/volume9/panait08a/panait08a.pdf). JMLR, 9, pp.423-457.
  * Schmidhuber, J., Wierstra, D., Gagliolo, M. and Gomez, F., 2007. [Training recurrent networks by evolino](https://direct.mit.edu/neco/article-abstract/19/3/757/7156/Training-Recurrent-Networks-by-Evolino). Neural Computation, 19(3), pp.757-779.
  * Gomez, F.J. and Schmidhuber, J., 2005, June. [Co-evolving recurrent neurons learn deep memory POMDPs](https://dl.acm.org/doi/10.1145/1068009.1068092). GECCO (pp. 491-498).
  * Fan, J., Lau, R. and Miikkulainen, R., 2003. [Utilizing domain knowledge in neuroevolution](https://www.aaai.org/Library/ICML/2003/icml03-025.php). ICML (pp. 170-177).
  * Potter, M.A. and De Jong, K.A., 2000. [Cooperative coevolution: An architecture for evolving coadapted subcomponents](https://direct.mit.edu/evco/article-abstract/8/1/1/859/Cooperative-Coevolution-An-Architecture-for). ECJ, 8(1), pp.1-29.
  * Gomez, F.J. and Miikkulainen, R., 1999, July. [Solving non-Markovian control tasks with neuroevolution](https://www.ijcai.org/Proceedings/99-2/Papers/097.pdf). IJCAI. (pp. 1356-1361).
  * Moriarty, D.E. and Mikkulainen, R., 1996. [Efficient reinforcement learning through symbiotic evolution](https://link.springer.com/article/10.1023/A:1018004120707). Machine Learning, 22(1), pp.11-32.
  * Moriarty, D.E. and Miikkulainen, R., 1995. [Efficient learning from delayed rewards through symbiotic evolution](https://www.sciencedirect.com/science/article/pii/B9781558603776500566). ICML (pp. 396-404). Morgan Kaufmann.
  * Potter, M.A. and De Jong, K.A., 1994, October. [A cooperative coevolutionary approach to function optimization](https://link.springer.com/chapter/10.1007/3-540-58484-6_269). PPSN (pp. 249-257). Springer, Berlin, Heidelberg.
* Simultaneous Perturbation Stochastic Approximation (SPSA) [ [https://www.jhuapl.edu/SPSA/](https://www.jhuapl.edu/SPSA/) ]
  * Spall, J.C., 2005. Introduction to stochastic search and optimization: Estimation, simulation, and control. John Wiley & Sons.
* Simulated Annealing (SA)
  * Bouttier, C. and Gavra, I., 2019. [Convergence rate of a simulated annealing algorithm with noisy observations](https://www.jmlr.org/papers/v20/16-588.html). JMLR, 20(1), pp.127-171.
  * Gerber, M. and Bornn, L., 2017. [Improving simulated annealing through derandomization](https://link.springer.com/article/10.1007/s10898-016-0461-1). JGO, 68(1), pp.189-217.
  * Siarry, P., Berthiau, G., Durdin, F. and Haussy, J., 1997. [Enhanced simulated annealing for globally minimizing functions of many-continuous variables](https://dl.acm.org/doi/abs/10.1145/264029.264043). TOMS, 23(2), pp.209-228.
  * Bertsimas, D. and Tsitsiklis, J., 1993. [Simulated annealing](https://tinyurl.com/yknunnpt). Statistical Science, 8(1), pp.10-15.
  * Corana, A., Marchesi, M., Martini, C. and Ridella, S., 1987. [Minimizing multimodal functions of continuous variables with the “simulated annealing” algorithm](https://dl.acm.org/doi/abs/10.1145/29380.29864). TOMS, 13(3), pp.262-280. [ [Corrigenda](https://dl.acm.org/doi/10.1145/66888.356281) ]
  * Kirkpatrick, S., Gelatt, C.D. and Vecchi, M.P., 1983. [Optimization by simulated annealing](https://science.sciencemag.org/content/220/4598/671). Science, 220(4598), pp.671-680.
  * Hastings, W.K., 1970. [Monte Carlo sampling methods using Markov chains and their applications](https://academic.oup.com/biomet/article/57/1/97/284580). Biometrika, 57(1), pp.97-109.
  * Metropolis, N., Rosenbluth, A.W., Rosenbluth, M.N., Teller, A.H. and Teller, E., 1953. [Equation of state calculations by fast computing machines](https://aip.scitation.org/doi/abs/10.1063/1.1699114). Journal of Chemical Physics, 21(6), pp.1087-1092.
  * Applications: e.g., [Young et al., 2023, Nature](https://www.nature.com/articles/s41586-023-05823-0); [Kim et al., 2023, Nature](https://www.nature.com/articles/s41586-023-06123-3); [Passalacqua et al., 2023, Nature](https://www.nature.com/articles/s41586-023-06229-8); [Pronker et al., 2023, Nature](https://www.nature.com/articles/s41586-023-06599-z); [Sullivan&Seljak, 2023](https://arxiv.org/pdf/2310.00745.pdf); [Holm et al., 2023, Nature](https://www.nature.com/articles/s41586-023-05908-w); [Snyder et al., 2023, Nature](https://www.nature.com/articles/s41586-022-05409-2); [Samyak&Palacios, 2023, Biometrika](https://academic.oup.com/biomet/advance-article/doi/10.1093/biomet/asad025/7143381); [Bouchet et al., 2023, PNAS](https://www.pnas.org/doi/abs/10.1073/pnas.2221407120); [Li&Yu, 2023, ACM-TOG](https://dl.acm.org/doi/10.1145/3592096); [Zhao et al., 2023, VLDBJ](https://link.springer.com/article/10.1007/s00778-023-00802-3); [Zhong et al., 2023, IEEE/ACM-TASLP](https://ieeexplore.ieee.org/abstract/document/10214657); [Wang et al., 2023, IEEE-TMC](https://ieeexplore.ieee.org/abstract/document/10011565); [Filippo et al., 2023, IJCAI](https://www.ijcai.org/proceedings/2023/0644.pdf); [Barnes et al., 2023, ApJ](https://iopscience.iop.org/article/10.3847/1538-4357/acba8e); [Melo et al., 2023](https://www.biorxiv.org/content/10.1101/2023.05.31.542906v1.full.pdf); [Bruna et al., 2023](https://www.biorxiv.org/content/biorxiv/early/2023/01/15/2023.01.13.524024.full.pdf); [Holm et al., 2023](https://arxiv.org/pdf/2309.04468.pdf); [Jenson et al., 2023, Nature](https://www.nature.com/articles/s41586-023-05862-7); [Kolesov et al., 2016, IEEE-TPAMI](https://ieeexplore.ieee.org/document/7130637)
* **Genetic Algorithm (GA)**
  * Whitley, D., 2019.
    [Next generation genetic algorithms: A user’s guide and tutorial](https://link.springer.com/chapter/10.1007/978-3-319-91086-4_8).
    In Handbook of Metaheuristics (pp. 245-274). Springer.
  * Levine, D., 1997.
    [Commentary—Genetic algorithms: A practitioner's view](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.9.3.256).
    IJOC, 9(3), pp.256-259.
  * Goldberg, D.E., 1994.
    [Genetic and evolutionary algorithms come of age](https://dl.acm.org/doi/10.1145/175247.175259).
    CACM, 37(3), pp.113-120.
  * Mitchell, M. and Forrest, S., 1994.
    [Genetic algorithms and artificial life](https://doi.org/10.1162/artl.1994.1.3.267)
    ALJ, 1(3), pp.267-289.
  * Forrest, S., 1993.
    [Genetic algorithms: Principles of natural selection applied to computation](https://www.science.org/doi/10.1126/science.8346439).
    Science, 261(5123), pp.872-878.
  * De Jong, K.A., 1993.
    [Are genetic algorithms function optimizer?](https://www.sciencedirect.com/science/article/pii/B9780080948324500064).
    FOGA, pp.5-17.
  * Mitchell, M., Holland, J. and Forrest, S., 1993.
    [When will a genetic algorithm outperform hill climbing](https://tinyurl.com/zfekmzhm).
    NeurIPS (pp. 51-58).
  * Whitley, D., Dominic, S., Das, R. and Anderson, C.W., 1993.
    [Genetic reinforcement learning for neurocontrol problems](https://tinyurl.com/5n6vvh8k).
    MLJ, 13, pp.259-284.
  * Holland, J.H., 1992.
    [Adaptation in natural and artificial systems: An introductory analysis with applications to
    biology, control, and artificial intelligence](https://tinyurl.com/ywm335f5).
    MIT Press.
  * Holland, J.H., 1992.
    [Genetic algorithms](https://www.scientificamerican.com/article/genetic-algorithms/).
    Scientific American, 267(1), pp.66-73.
  * Whitley, D., 1989, December.
    [The GENITOR algorithm and selection pressure: Why rank-based allocation of reproductive trials is best](https://dl.acm.org/doi/10.5555/93126.93169).
    FOGA (pp. 116-121).
  * Goldberg, D.E. and Holland, J.H., 1988.
    [Genetic algorithms and machine learning](https://link.springer.com/article/10.1023/A:1022602019183).
    MLJ, 3(2), pp.95-99.
  * Holland, J.H., 1973.
    [Genetic algorithms and the optimal allocation of trials](https://epubs.siam.org/doi/10.1137/0202009).
    SICOMP, 2(2), pp.88-105.
  * Holland, J.H., 1962.
    [Outline for a logical theory of adaptive systems](https://dl.acm.org/doi/10.1145/321127.321128).
    JACM, 9(3), pp.297-314.
  * Applications: e.g., [Wang, 2023, Harvard Ph.D. Dissertation](https://dash.harvard.edu/bitstream/handle/1/37374599/dissertation.pdf); [Lee et al., 2022, Science Robotics](https://www.science.org/doi/10.1126/scirobotics.abq7278); [Whitelam&Tamblyn, 2021, PRL](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.018003); [Walker et al., 2021, Nature Chemistry](https://www.nature.com/articles/s41557-020-00626-6); [Chen et al., 2020, Nature](https://www.nature.com/articles/s41586-019-1901-0).
* Evolutionary Programming (EP)
  * Yao, X., Liu, Y. and Lin, G., 1999. [Evolutionary programming made faster](https://ieeexplore.ieee.org/abstract/document/771163). TEVC, 3(2), pp.82-102.
  * Fogel, D.B., 1999. [An overview of evolutionary programming](https://link.springer.com/chapter/10.1007/978-1-4612-1542-4_5). In Evolutionary Algorithms (pp. 89-109). Springer, New York, NY.
  * Fogel, D.B. and Fogel, L.J., 1995, September. [An introduction to evolutionary programming](https://link.springer.com/chapter/10.1007/3-540-61108-8_28). In European Conference on Artificial Evolution (pp. 21-33). Springer, Berlin, Heidelberg.
  * Fogel, D.B., 1994. [Evolutionary programming: An introduction and some current directions](https://link.springer.com/article/10.1007/BF00175356). Statistics and Computing, 4(2), pp.113-129.
  * Bäck, T. and Schwefel, H.P., 1993.
    [An overview of evolutionary algorithms for parameter optimization](https://direct.mit.edu/evco/article-abstract/1/1/1/1092/An-Overview-of-Evolutionary-Algorithms-for).
    ECJ, 1(1), pp.1-23.
  * Fogel, L.J., Owens, A.J. and Walsh, M.J., 1965.
    Intelligent decision making through a simulation of evolution.
    IEEE Transactions on Human Factors in Electronics, 6(1), pp.13-23.
### **Pattern/Direct Search (PS/DS)**
  * Porcelli, M. and [Toint, P.L.](https://researchportal.unamur.be/en/persons/phtoint), 2022.
    [Exploiting problem structure in derivative free optimization](https://dl.acm.org/doi/abs/10.1145/3474054).
    TOMS, 48(1), pp.1-25.
  * Audet, C., Le Digabel, S., Montplaisir, V.R. and Tribes, C., 2022.
    [Algorithm 1027: NOMAD version 4: Nonlinear optimization with the MADS algorithm](https://dl.acm.org/doi/abs/10.1145/3544489).
    TOMS, 48(3), pp.1-22.
  * Singer, S. and Nelder, J., 2009.
    [Nelder-mead algorithm](http://var.scholarpedia.org/article/Nelder-Mead_algorithm).
    Scholarpedia, 4(7), p.2928.
  * Dolan, E.D., Lewis, R.M. and Torczon, V., 2003.
    [On the local convergence of pattern search]().
    SIOPT, 14(2), pp.567-583.
  * Lagarias, J.C., Reeds, J.A., Wright, M.H. and Wright, P.E., 1998.
    [Convergence properties of the Nelder--Mead simplex method in low dimensions](https://epubs.siam.org/doi/abs/10.1137/S1052623496303470).
    SIOPT, 9(1), pp.112-147.
  * Powell, M.J., 1998.
    [Direct search algorithms for optimization calculations](https://www.cambridge.org/core/journals/acta-numerica/article/abs/direct-search-algorithms-for-optimization-calculations/23FA5B19EAF122E02D3724DB1841238C).
    AN, 7, pp.287-336.
  * McKinnon, K.I., 1998.
    Convergence of the Nelder--Mead simplex method to a nonstationary point.
    SIOPT, 9(1), pp.148-158.
  * Torczon, V., 1997.
    [On the convergence of pattern search algorithms](https://epubs.siam.org/doi/abs/10.1137/S1052623493250780).
    SIOPT, 7(1), pp.1-25.
  * Barton, R.R. and Ivey Jr, J.S., 1996.
    [Nelder-Mead simplex modifications for simulation optimization](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.42.7.954).
    MS, 42(7), pp.954-973.
  * [Wright, M.H.](https://www.simonsfoundation.org/people/margaret-h-wright/), 1996.
    [Direct search methods: Once scorned, now respectable](https://nyuscholars.nyu.edu/en/publications/direct-search-methods-once-scorned-now-respectable).
    Pitman Research Notes in Mathematics Series, pp.191-208.
  * Powell, M.J., 1973.
    [On search directions for minimization algorithms](https://link.springer.com/article/10.1007/BF01584660).
    Mathematical Programming, 4(1), pp.193-201.
  * Nelder, J.A. and Mead, R., 1965.
    [A simplex method for function minimization](https://academic.oup.com/comjnl/article-abstract/7/4/308/354237).
    CJ, 7(4), pp.308-313.
  * Powell, M.J., 1964.
    [An efficient method for finding the minimum of a function of several variables without calculating derivatives](https://academic.oup.com/comjnl/article-abstract/7/2/155/335330).
    CJ, 7(2), pp.155-162.
  * Kaupe Jr, A.F., 1963.
    [Algorithm 178: Direct search](https://dl.acm.org/doi/pdf/10.1145/366604.366632).
    CACM, 6(6), pp.313-314.
  * Hooke, R. and Jeeves, T.A., 1961.
    [“Direct search” solution of numerical and statistical problems](https://dl.acm.org/doi/10.1145/321062.321069).
    JACM, 8(2), pp.212-229.
  * [Fermi, E.](https://www.nobelprize.org/prizes/physics/1938/fermi/biographical/) and
    [Metropolis N.](https://history.computer.org/pioneers/metropolis.html), 1952.
    [Numerical solution of a minimum problem](https://www.osti.gov/servlets/purl/4377177).
    TR, Los Alamos Scientific Lab.
### **Random Search/Optimization (RS/RO)**
  * Wang, X., Hong, L.J., Jiang, Z. and Shen, H., 2025.
    [Gaussian process-based random search for continuous optimization via simulation](https://pubsonline.informs.org/doi/abs/10.1287/opre.2021.0303).
    OR, 73(1), pp.385-407.
  * Sel, B., et al., 2023, June.
    [Learning-to-learn to guide random search: Derivative-free meta blackbox optimization on manifold](https://proceedings.mlr.press/v211/sel23a.html).
    In L4DC (pp. 38-50).
  * Gao, K. and Sener, O., 2022.
    [Generalizing Gaussian smoothing for random search](https://proceedings.mlr.press/v162/gao22f.html).
    ICML (pp. 7077-7101).
  * Sener, O. and Koltun, V., 2020.
    [Learning to guide random search](https://openreview.net/forum?id=B1gHokBKwS).
    In ICLR.
  * [Nesterov, Y.](https://www.nasonline.org/directory-entry/yurii-e-nesterov-5n5mo7/) and Spokoiny, V., 2017.
    [Random gradient-free minimization of convex functions](https://link.springer.com/article/10.1007/s10208-015-9296-2).
    FoCM, 17(2), pp.527-566.
  * [Stich, S.U.](), 2014.
    [On low complexity acceleration techniques for randomized optimization](https://link.springer.com/chapter/10.1007/978-3-319-10762-2_13).
    In PPSN (pp. 130-140). Springer.
  * Appel, M.J., Labarre, R. and Radulovic, D., 2004.
    [On accelerated random search](https://epubs.siam.org/doi/abs/10.1137/S105262340240063X).
    SIOPT, 14(3), pp.708-731.
  * [Schmidhuber, J.](), [Hochreiter, S.]() and [Bengio, Y.](), 2001.
    [Evaluating benchmark problems by random guessing](https://ml.jku.at/publications/older/ch9.pdf).
    A Field Guide to Dynamical Recurrent Networks, pp.231-235.
  * Rosenstein, M.T. and [Barto, A.G.](https://people.cs.umass.edu/~barto/), 2001.
    [Robot weightlifting by direct policy search](https://dl.acm.org/doi/abs/10.5555/1642194.1642206).
    IJCAI. (pp. 839-846).
  * Sarma, M.S., 1990.
    [On the convergence of the Baba and Dorea random optimization methods](https://link.springer.com/article/10.1007/BF00939542).
    JOTA, 66, pp.337-343.
  * [Polyak, B.T.](https://sites.google.com/site/lab7polyak/), 1987.
    [Introduction to optimization](https://sites.google.com/site/lab7polyak/).
    Optimization Software Inc.
  * Rastrigin, L.A., 1986.
    [Random search as a method for optimization and adaptation](https://link.springer.com/chapter/10.1007/BFb0007129).
    In Stochastic Optimization. Springer.
  * Dorea, C.C.Y., 1983.
    [Expected number of steps of a random optimization method](https://link.springer.com/article/10.1007/BF00934526).
    JOTA, 39(2), pp.165-171.
  * Baba, N., 1981.
    [Convergence of a random optimization method for constrained optimization problems](https://link.springer.com/article/10.1007/BF00935752).
    JOTA, 33(4), pp.451-461.
  * Solis, F.J. and Wets, R.J.B., 1981.
    [Minimization by random search techniques](https://pubsonline.informs.org/doi/abs/10.1287/moor.6.1.19).
    MOR, 6(1), pp.19-30.
  * Schumer, M.A. and [Steiglitz, K.](https://www.cs.princeton.edu/~ken/), 1968.
    [Adaptive step size random search](https://ieeexplore.ieee.org/abstract/document/1098903).
    TAC, 13(3), pp.270-276.
  * Matyas, J., 1965.
    [Random optimization](https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=at&paperid=11288).
    ARC, 26(2), pp.246-253.
  * Rastrigin, L.A., 1963.
    [The convergence of the random search method in the extremal control of a many parameter system](https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=at&paperid=12312).
    ARC, 24, pp.1337-1342.
  * Brooks, S.H., 1958.
    [A discussion of random methods for seeking maxima](https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244).
    OR, 6(2), pp.244-251.
  * [Ashby, W.R.](https://ashby.info), 1952.
    [Design for a brain: The origin of adaptive behaviour](https://link.springer.com/book/10.1007/978-94-015-1320-3).
    Springer.

## Pioneers of Evolutionary and Swarm-Based Optimization

* [Box, G.E.](https://royalsocietypublishing.org/doi/pdf/10.1098/rsbm.2015.0015), 1957.
  [Evolutionary operation: A method for increasing industrial productivity](https://rss.onlinelibrary.wiley.com/doi/abs/10.2307/2985505).
  JRSS-SC, 6(2), pp.81-101.

## Financial Sponsors

From 2021 to 2023, this open-source pure-Python library was supported by
**Shenzhen Fundamental Research Program** (a total of 2 000 000 Yuan).

## Citations and Activities

If this open-source Python library is used in your paper or project,
it is highly welcomed *but NOT mandatory* to cite the following
arXiv [preprint](https://arxiv.org/abs/2212.05652) paper (First
Edition: 12 Dec 2022): **Duan, Q., Zhou, G., Shao, C., and Others,
2024. PyPop7: A Pure-Python Library for Population-Based Black-Box
Optimization. arXiv preprint arXiv:2212.05652.** (Now it has been
submitted to [JMLR](https://jmlr.org/), after 3-round reviews from
28 Mar 2023 to 01 Nov 2023 to 05 Jul 2024, and finally accepted in
11 Oct 2024.)

### 2026 (Planned Activities)

* A **tutorial** on Open-Source Software for Swarm Intelligence and 
  Evolutionary Algorithms is planning for [ICSI 2026].
* A **tutorial** on Open-Source Software for Evolutionary Algorithms
  and Swarm Intelligence is planning for [IEEE-SSCI 2026].

### 2025 (Activities)

* A 20-minute online report will be given by one author
  (Qiqi Duan in 13 Sept 2025) in Chinese for
  [LEAD (Workshop on Learning-assisted Evolutionary Algorithm Design)](https://sites.google.com/view/leadworkshop2025/).
  * Oral: [ppt (in English)]()
* A presentation was given by one author (Qiqi Duan in
  11 Jun 2025) in [IEEE-CEC 2025]().
  * Oral: [ppt]().
* A one-page paper was submitted to [IEEE-CEC 2025]() as
  one *Journal-to-Conference* ([J2C]()) paper and was
  accepted in 18 Mar 2025.
  * J2C: [pdf](https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/presentations/IEEE-CEC-2025-%5BJ2C%5D.pdf).

### BibTeX (2024)

@article{JMLR-2024-Duan,
  title={{PyPop7}: A {pure-Python} library for population-based black-box optimization},
  author={Duan, Qiqi and Zhou, Guochen and Shao, Chang and Others},
  journal={Journal of Machine Learning Research},
  volume={25},
  number={296},
  pages={1--28},
  year={2024}
}

## Visitors and Stars

![visitors](https://tinyurl.com/4cu8wn4u)

[![Star](https://tinyurl.com/yy7pfjwz)](https://tinyurl.com/w9wwc54f)
