# pypop

[![GNU General Public License v3.0](https://img.shields.io/badge/license-GNU%20GPL--v3.0-green.svg)](https://github.com/Evolutionary-Intelligence/pypop/blob/main/LICENSE) [![gitter for pypop](https://img.shields.io/badge/gitter-pypop--go-brightgreen.svg)](https://gitter.im/pypop-go/community)

```PyPop``` is a *Pure-PYthon* library of **POPulation-based OPtimization** for real-parameter, black-box problems (**currently actively developed**). Its goal is to provide a *unified* interface and also *elegant* implementations for **Derivative-Free Optimization (DFO)**, *particularly population-based optimizers*, in order to facilitate research repeatability and also real-world applications.

<p align="center">
<img src="https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/logo/PyPop-Logo-Small-0.png" alt="drawing" width="321"/>
</p>

For alleviating the notorious **curse of dimensionality** of DFO (based on *iterative sampling*), the main focus of ```PyPop``` is to cover their **State-Of-The-Art implementations for Large-Scale Optimization (LSO)**, though their other versions and variants may be also included here (e.g., for benchmarking purpose, for mixing purpose, and sometimes even for practical purpose).

## A (*Growing*) List of **Publically Available** Gradient-Free Optimizers (GFO)

* **Natural Evolution Strategies (NES)** [See e.g. [Wierstra et al., 2014, JMLR](https://jmlr.org/papers/v15/wierstra14a.html)]

  * Rank-One Natural Evolution Strategy (**R1NES**) [See [Sun et al., 2013, GECCO](https://dl.acm.org/doi/abs/10.1145/2464576.2464608)]

  * Separable Natural Evolution Strategy (**SNES**) [See [Schaul et al., 2011, GECCO](https://dl.acm.org/doi/abs/10.1145/2001576.2001692)]

* **Estimation of Distribution Algorithms** (**EDA**)

* **Particle Swarm Optimization (PSO)** [See e.g. [Kennedy and Eberhart, 1995, ICNN](https://ieeexplore.ieee.org/document/488968)]

* **CoOperative co-Evolutionary Algorithms (COEA)** [See e.g. [Gomez et al., 2008, JMLR](https://jmlr.org/papers/v9/gomez08a.html); [Panait et al., 2008, JMLR](https://www.jmlr.org/papers/v9/panait08a.html)]

  * CoOperative SYnapse NeuroEvolution (**COSYNE / CoSyNE**) [See [Gomez et al., 2008, JMLR](https://jmlr.org/papers/v9/gomez08a.html)]

* **Simulated Annealing (SA)** [See e.g. [Kirkpatrick et al., 1983, Science](https://www.science.org/doi/10.1126/science.220.4598.671)]

  * [![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg)](https://github.com/Evolutionary-Intelligence/pypop/edit/main/README.md) Enhanced Simulated Annealing (**ESA**) [See [Siarry et al., 1997, ACM-TOMS](https://dl.acm.org/doi/abs/10.1145/264029.264043)]

  * [![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg)](https://github.com/Evolutionary-Intelligence/pypop/edit/main/README.md) Corana et al.' Simulated Annealing (**CSA**) [See [Corana et al., 1987, ACM-TOMS](https://dl.acm.org/doi/abs/10.1145/29380.29864)]

* **Evolution Strategies (ES)** [See e.g. [Beyer and Schwefel, 2002, Natural Computing](https://link.springer.com/article/10.1023/A:1015059928466); [Schwefel, 1984, Ann Oper Res](https://link.springer.com/article/10.1007/BF01876146)]

  * Fast Matrix Adaptation Evolution Strategy (**FMAES**, Fast-(μ/μ_w, λ)-MA-ES) [See [Beyer, 2020, GECCO](https://dl.acm.org/doi/abs/10.1145/3377929.3389870)]

  * Rank-m Evolution Strategy with Multiple Evolution Paths (**RMES / RmES**) [See [Li and Zhang, 2018, TEVC](https://ieeexplore.ieee.org/document/8080257)]

  * Rank-One Evolution Strategy (**R1ES**) [See [Li and Zhang, 2018, TEVC](https://ieeexplore.ieee.org/document/8080257)]

  * [![competitor](https://img.shields.io/badge/**-competitor-blue.svg)](https://github.com/Evolutionary-Intelligence/pypop/edit/main/README.md) Matrix Adaptation Evolution Strategy (**MAES**, (μ/μ_w, λ)-MA-ES) [See [Beyer and Sendhoff, 2017, TEVC](https://ieeexplore.ieee.org/abstract/document/7875115/)]

  * [![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg)](https://github.com/Evolutionary-Intelligence/pypop/edit/main/README.md) Self-Adaptation Evolution Strategy (**SAES**, (μ/μ_I, λ)-σSA-ES) [See e.g. [Beyer, 2020, GECCO](https://dl.acm.org/doi/abs/10.1145/3377929.3389870); [Beyer, 2007, Scholarpedia](http://www.scholarpedia.org/article/Evolution_strategies)]

* **Genetic Algorithms (GA)** [See e.g. [Holland, 1962, JACM](https://dl.acm.org/doi/10.1145/321127.321128)]

* **Evolutionary Programming (EP)** [See e.g. [Yao et al., 1999, TEVC](https://ieeexplore.ieee.org/abstract/document/771163)]

* **Random (Stochastic) Search (RS)** [See e.g. [Brooks, 1958, Operations Research](https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244)]

  * [![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg)](https://github.com/Evolutionary-Intelligence/pypop/edit/main/README.md) Pure Random Search (**PRS**) [See e.g. [Bergstra and Bengio, 2012, JMLR](https://www.jmlr.org/papers/v13/bergstra12a.html)]

  * [![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg)](https://github.com/Evolutionary-Intelligence/pypop/edit/main/README.md) Random Hill Climber (**RHC**) [See e.g. [Schaul et al., 2010, JMLR](https://jmlr.org/papers/v11/schaul10a.html)]
  
  * [![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg)](https://github.com/Evolutionary-Intelligence/pypop/edit/main/README.md) Annealed Random Hill Climber (**ARHC**) [See e.g. [Schaul et al., 2010, JMLR](https://jmlr.org/papers/v11/schaul10a.html)]

## Design Philosophy

* **Respect for Diversity**

  * Given the universality of black-box optimization in science and engineering, different research communities have designed different methods, till now. On the one hand, some of these methods may share similarities more or less. On the other hand, these methods may show significant differences. As a result, we hope to cover such a diversity from different research communities such as artificial intelligence (particularly evolutionary computation and machine learning (zeroth-order optimization)), mathematical optimization (particularly global optimization), operations research, etc..

* **Respect for Originality**

  * *“It is both enjoyable and educational to hear the ideas directly from the creators”.* (From Hennessy, J.L. and Patterson, D.A., 2019. Computer architecture: A quantitative approach (Sixth Edition). Elsevier.)

  * For each optimizer considered here, we expect to give its original/representative reference (including its good implementations/improvements). If you find some important reference missed here, please DO NOT hesitate to contact us (we will be happy to add it if necessary).

* **Respect for Repeatability**

  * For randomized search, properly controlling randomness is very crucial to repeat numerical experiments. Here we follow the *Random Sampling* suggestions from [NumPy](https://numpy.org/doc/stable/reference/random/). In other worlds, you must **explicitly** set the random seed for each optimizer.

## Research Support

This open-source Python library for black-box optimization is now supported by **Shenzhen Fundamental Research Program** under Grant No. JCYJ20200109141235597 (￥2,000,000), granted to **Prof. Yuhui Shi** (CSE, SUSTech @ Shenzhen, China), and actively developed (from 2021 to 2023) by his group members (e.g., *Qiqi Duan*, *Chang Shao*, *Guochen Zhou*, and Youkui Zhang).

We also acknowledge the initial discussions and Python code efforts (dating back to *about* 2017) with Hao Tong from the research group of **Prof. Yao** (CSE, SUSTech @ Shenzhen, China).
