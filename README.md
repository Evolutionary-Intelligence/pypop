# pypop7 (Pure-PYthon library of POPulation-based black-box OPtimization)

[![GNU General Public License v3.0](https://img.shields.io/badge/license-GNU%20GPL--v3.0-green.svg)](https://github.com/Evolutionary-Intelligence/pypop/blob/main/LICENSE) [![gitter for pypop](https://img.shields.io/badge/gitter-pypop--go-brightgreen.svg)](https://gitter.im/pypop-go/community) [![PyPI for pypop7](https://img.shields.io/badge/PyPI-pypop7-yellowgreen.svg)](https://pypi.org/project/pypop7/)

```PyPop7``` is a *Pure-PYthon* library of **POPulation-based OPtimization** for single-objective, real-parameter, black-box problems (**currently actively developed**). Its main goal is to provide a *unified* interface and *elegant* implementations for **Derivative-Free Optimization (DFO)**, *particularly population-based optimizers*, in order to facilitate research repeatability and also real-world applications.

<p align="center">
<img src="https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/logo/PyPop-Logo-Small-0.png" alt="drawing" width="321"/>
</p>

More specifically, for alleviating the notorious **curse of dimensionality** of DFO (based on *iterative sampling*), the primary focus of ```PyPop7``` is to cover their **State-Of-The-Art (SOTA) implementations for Large-Scale Optimization (LSO)**, though many of their other versions and variants are also included here (for benchmarking/mixing purpose, and sometimes even for practical purpose).

## How to Use PyPop7

The following three simple steps are enough to utilize the optimization power of [PyPop7](https://pypi.org/project/pypop7/):

1. Use [pip](https://pypi.org/project/pip/) to install ```pypop7``` on the virtual environment via [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```bash
$ pip install pypop7
```

2. Define your own objective function for the optimization problem at hand:

```Python
import numpy as np  # for numerical computation, which is also the computing engine of pypop7

# the below example is Rosenbrock, the notorious test function in the optimization community
def rosenbrock(x):
    return 100 * np.sum(np.power(x[1:] - np.power(x[:-1], 2), 2)) + np.sum(np.power(x[:-1] - 1, 2))

# define the fitness (cost) function and also its settings
ndim_problem = 1000
problem = {'fitness_function': rosenbrock,  # cost function
           'ndim_problem': ndim_problem,  # dimension
           'lower_boundary': -5 * np.ones((ndim_problem,)),  # search boundary
           'upper_boundary': 5 * np.ones((ndim_problem,))}
```

3. Run one or more black-box optimizers from ```pypop7``` on the given optimization problem:

```Python
# all available optimizers can be seen in
#    https://github.com/Evolutionary-Intelligence/pypop/tree/main/pypop7/optimizers
# here we choose LM-MA-ES owing to its low complexity and metric-learning ability
from pypop7.optimizers.es.lmmaes import LMMAES
# define all the necessary algorithm options (which differ among different optimizers)
options = {'fitness_threshold': 1e-10,
           'max_runtime': 3600,  # 1 hours
           'seed_rng': 0,
           'x': 4 * np.ones((ndim_problem,)),  # initial mean of search (mutation) distribution
           'sigma': 0.3,  # initial global step-size of search (mutation) distribution
           'verbose_frequency': 500}
lmmaes = LMMAES(problem, options)  # initilize the optimizer
results = lmmaes.optimize()  # run its search process
print(results)
```

Below a [DEMO](https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/demo/demo.py) is given on a toy 2-dimensional minimization function, in order to visually show the very interesting/powerful evolutionary search process of [```MAES```](https://ieeexplore.ieee.org/abstract/document/7875115/) (i.e., Covariance Matrix Adaptation):

<p align="center">
<img src="https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/demo/demo.gif" alt="drawing" width="432"/>
</p>

## A (*Still Growing*) List of **Publicly Available** Gradient-Free Optimizers (GFO)

******* *** *******

![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg): indicates the specific version for LSO (e.g., dimension >= 1000).

![competitor](https://img.shields.io/badge/**-competitor-blue.svg): indicates the competitive (or *de facto*) version for *relatively low-dimensional* problems (though it may also work well under certain LSO circumstances).

![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg): indicates the baseline version for benchmarking purpose or for theoretical interest.

******* *** *******

* **Evolution Strategies (ES)** [See e.g. [Ollivier et al., 2017, JMLR](https://www.jmlr.org/papers/v18/14-467.html); [Hansen et al., 2015](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44); [Bäck et al., 2013](https://link.springer.com/book/10.1007/978-3-642-40137-4); [Rudolph, 2012](https://link.springer.com/referenceworkentry/10.1007/978-3-540-92910-9_22); [Beyer&Schwefel, 2002](https://link.springer.com/article/10.1023/A:1015059928466); [Rechenberg, 1989](https://link.springer.com/chapter/10.1007/978-3-642-83814-9_6); [Schwefel, 1984](https://link.springer.com/article/10.1007/BF01876146)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Mixture Model-based Evolution Strategy (**[MMES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/mmes.py)**) [See [He et al., 2021, TEVC](https://ieeexplore.ieee.org/abstract/document/9244595)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Limited-Memory Matrix Adaptation Evolution Strategy (**[LMMAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/lmmaes.py)**, LM-MA-ES) [See [Loshchilov et al., 2019, TEVC](https://ieeexplore.ieee.org/abstract/document/8410043)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Rank-m Evolution Strategy *with Multiple Evolution Paths* (**[RMES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/rmes2.py)**, Rm-ES) [See [Li&Zhang, 2018, TEVC](https://ieeexplore.ieee.org/document/8080257)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Rank-One Evolution Strategy (**[R1ES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/r1es.py)**, R1-ES) [See [Li&Zhang, 2018, TEVC](https://ieeexplore.ieee.org/document/8080257)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Projection-based Covariance Matrix Adaptation (**VKDCMA**, VkD-CMA) [See [Akimoto&Hansen, 2016, GECCO](https://dl.acm.org/doi/abs/10.1145/2908812.2908863)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Linear Covariance Matrix Adaptation (**[VDCMA](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/vdcma.py)**, VD-CMA) [See [Akimoto et al., 2014, GECCO](https://dl.acm.org/doi/abs/10.1145/2576768.2598258)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Cholesky-CMA-ES-2009 (**[CCMAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/ccmaes.py)**) [See [Suttorp et al., 2009, MLJ](https://link.springer.com/article/10.1007/s10994-009-5102-1)]
 
    * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) (1+1)-Cholesky-CMA-ES-2009 (**[OPOC2009](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/opoc2009.py)**) [See [Suttorp et al., 2009, MLJ](https://link.springer.com/article/10.1007/s10994-009-5102-1)]

    * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) (1+1)-Cholesky-CMA-ES (**[OPOC](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/opoc.py)**) [See [Igel et al., 2006, GECCO](https://dl.acm.org/doi/abs/10.1145/1143997.1144082)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Separable Covariance Matrix Adaptation Evolution Strategy (**[SEPCMAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/optimizers/es/sepcmaes.py)**, sep-CMA-ES) [See [Bäck et al., 2013](https://link.springer.com/book/10.1007/978-3-642-40137-4); [Ros&Hansen, 2008, PPSN](https://link.springer.com/chapter/10.1007/978-3-540-87700-4_30)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Main Vector Adaptation Evolution Strategies (**[MVAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/mvaes.py)**, MVA-ES) [See [Poland&Zell, 2001, GECCO](https://dl.acm.org/doi/abs/10.5555/2955239.2955428)]

  * ![competitor](https://img.shields.io/badge/**-competitor-blue.svg) Diagonal Decoding Covariance Matrix Adaptation (**DDCMA**, dd-CMA) [See [Akimoto&Hansen, 2019, ECJ](https://direct.mit.edu/evco/article/28/3/405/94999/Diagonal-Acceleration-for-Covariance-Matrix)]

  * ![competitor](https://img.shields.io/badge/**-competitor-blue.svg) Covariance Matrix Self-Adaptation with Repelling Subpopulations (**RSCMSA**, RS-CMSA) [See [Ahrari et al., 2017, ECJ](https://doi.org/10.1162/evco_a_00182)]

  * ![competitor](https://img.shields.io/badge/**-competitor-blue.svg) Matrix Adaptation Evolution Strategy (**[MAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/optimizers/es/maes.py)**, (μ/μ_w,λ)-MA-ES) [See [Beyer&Sendhoff, 2017, TEVC](https://ieeexplore.ieee.org/abstract/document/7875115/)]

    * ![competitor](https://img.shields.io/badge/**-competitor-blue.svg) Fast Matrix Adaptation Evolution Strategy (**[FMAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/optimizers/es/fmaes.py)**, Fast-(μ/μ_w,λ)-MA-ES) [See [Beyer, 2020, GECCO](https://dl.acm.org/doi/abs/10.1145/3377929.3389870); [Loshchilov et al., 2019, TEVC](https://ieeexplore.ieee.org/abstract/document/8410043)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Self-Adaptation Evolution Strategy (**SAES**, (μ/μ_I, λ)-σSA-ES) [See e.g. [Beyer, 2020, GECCO](https://dl.acm.org/doi/abs/10.1145/3377929.3389870); [Beyer, 2007, Scholarpedia](http://www.scholarpedia.org/article/Evolution_strategies)]

    * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Cumulative Step-size Adaptation Evolution Strategy (**CSAES**, (μ/μ,λ)-ES)  [See e.g. [Hansen et al., 2015](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44); [Ostermeier et al., 1994, PPSN](https://link.springer.com/chapter/10.1007/3-540-58484-6_263)]

    * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Derandomized Self-Adaptation Evolution Strategy (**DSAES**, (1,λ)-σSA-ES) [See e.g. [Hansen et al., 2015](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44); [Ostermeier et al., 1994, ECJ](https://direct.mit.edu/evco/article-abstract/2/4/369/1407/A-Derandomized-Approach-to-Self-Adaptation-of)]

    * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Schwefel's Self-Adaptation Evolution Strategy (**[SSAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/optimizers/es/ssaes.py)**, (μ/μ,λ)-σSA-ES) [See e.g. [Hansen et al., 2015](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44)]

    * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Rechenberg's (1+1)-Evolution Strategy with 1/5th success rule (**[RES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/res.py)**) [See e.g. [Hansen et al., 2015](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44); [Kern et al., 2004](https://link.springer.com/article/10.1023/B:NACO.0000023416.59689.4e); [Schumer&Steiglitz, 1968, IEEE-TAC](https://ieeexplore.ieee.org/abstract/document/1098903)]

* **Natural Evolution Strategies (NES)** [See e.g. [Wierstra et al., 2014, JMLR](https://jmlr.org/papers/v15/wierstra14a.html); [Yi et al., 2009, ICML](https://dl.acm.org/doi/abs/10.1145/1553374.1553522); [Wierstra et al., 2008, CEC](https://ieeexplore.ieee.org/document/4631255)]

  * Rank-One Natural Evolution Strategy (**R1NES**) [See [Sun et al., 2013, GECCO](https://dl.acm.org/doi/abs/10.1145/2464576.2464608)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Separable Natural Evolution Strategy (**SNES**) [See [Schaul et al., 2011, GECCO](https://dl.acm.org/doi/abs/10.1145/2001576.2001692)]

* **Estimation of Distribution Algorithms** (**EDA**) [See e.g. [Larrañaga&Lozano, 2002](https://link.springer.com/book/10.1007/978-1-4615-1539-5); [Pelikan et al., 2002](https://link.springer.com/article/10.1023/A:1013500812258); [Mühlenbein&Paaß, 1996, PPSN](https://link.springer.com/chapter/10.1007/3-540-61723-X_982)]

* **Cross-Entropy Method** (**CEM**) [See e.g. [Rubinstein&Kroese, 2004](https://link.springer.com/book/10.1007/978-1-4757-4321-0)]

* **Particle Swarm Optimization (PSO)** [See e.g. [Shi&Eberhart, 1998, CEC](https://ieeexplore.ieee.org/abstract/document/699146); [Kennedy&Eberhart, 1995, ICNN](https://ieeexplore.ieee.org/document/488968)]

* **CoOperative co-Evolutionary Algorithms (COEA)** [See e.g. [Gomez et al., 2008, JMLR](https://jmlr.org/papers/v9/gomez08a.html); [Panait et al., 2008, JMLR](https://www.jmlr.org/papers/v9/panait08a.html)]

  * CoOperative SYnapse NeuroEvolution (**COSYNE**, CoSyNE) [See [Gomez et al., 2008, JMLR](https://jmlr.org/papers/v9/gomez08a.html)]

* **Simulated Annealing (SA)** [See e.g. [Kirkpatrick et al., 1983, Science](https://www.science.org/doi/10.1126/science.220.4598.671); [Hastings, 1970, Biometrika](https://academic.oup.com/biomet/article/57/1/97/284580); [Metropolis et al., 1953, JCP](https://aip.scitation.org/doi/abs/10.1063/1.1699114)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Enhanced Simulated Annealing (**ESA**) [See [Siarry et al., 1997, ACM-TOMS](https://dl.acm.org/doi/abs/10.1145/264029.264043)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Corana et al.' Simulated Annealing (**CSA**) [See [Corana et al., 1987, ACM-TOMS](https://dl.acm.org/doi/abs/10.1145/29380.29864)]

* **Genetic Algorithms (GA)** [See e.g. [Forrest, 1993, Science](https://www.science.org/doi/abs/10.1126/science.8346439); [Holland, 1962, JACM](https://dl.acm.org/doi/10.1145/321127.321128)]

* **Evolutionary Programming (EP)** [See e.g. [Yao et al., 1999, TEVC](https://ieeexplore.ieee.org/abstract/document/771163)]

* **Differential Evolution (DE)** [See e.g. [Storn&Price, 1997, JGO](https://link.springer.com/article/10.1023/A:1008202821328)]

* **Direct Search (DS)** [See e.g. [Wright, 1996](https://nyuscholars.nyu.edu/en/publications/direct-search-methods-once-scorned-now-respectable); [Hooke&Jeeves, 1961, JACM](https://dl.acm.org/doi/10.1145/321062.321069)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Nelder-Mead Simplex Method (NelderMead) [See [Nelder&Mead, 1965, Computer](https://academic.oup.com/comjnl/article-abstract/7/4/308/354237)]

* **Random (Stochastic) Search (RS)** [See e.g. [Rastrigin, 1986](https://link.springer.com/content/pdf/10.1007/BFb0007129.pdf); [Brooks, 1958, Operations Research](https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Pure Random Search (**PRS**) [See e.g. [Bergstra and Bengio, 2012, JMLR](https://www.jmlr.org/papers/v13/bergstra12a.html)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Random Hill Climber (**RHC**) [See e.g. [Schaul et al., 2010, JMLR](https://jmlr.org/papers/v11/schaul10a.html)]
  
  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Annealed Random Hill Climber (**ARHC**) [See e.g. [Schaul et al., 2010, JMLR](https://jmlr.org/papers/v11/schaul10a.html)]

## Design Philosophy

* **Respect for Beauty (Elegance)**

  * From the *problem-solving* perspective, we empirically prefer to choose the *best* optimizer for the black-box optimization problem at hand. However, for the *new* problem, the *best* optimizer is often unknown in advance (without *a prior* knowledge). As a rule of thumb, we need to compare a (often small) set of all available/well-known optimizers and choose the *best* one from them according to some predefined performance criteria. From the *research* perspective, however, we like *beautiful* optimizers, though always keeping the **[“No Free Lunch” theorem](https://ieeexplore.ieee.org/document/585893)** in mind. Typically, the **beauty** of one optimizer comes from the following features: **novelty** (e.g., GA/PSO), **competitive performance** on at least one class of problems (e.g., BO), **theoretical insights** (e.g., CMA-ES/NES), **clarity/simplicity** (e.g., CEM/EDA), and **repeatability**.

  * If you find any DFO to meet the above standard, welcome to launch [issues](https://github.com/Evolutionary-Intelligence/pypop/issues) or [pulls](https://github.com/Evolutionary-Intelligence/pypop/pulls). We will consider it to be included in the ```pypop``` library. Note that **[any superficial imitation](https://dl.acm.org/doi/10.1145/3402220.3402221)** to the above well-established optimizers (**['Old Wine in a New Bottle'](https://link.springer.com/article/10.1007/s11721-021-00202-9)**) will be *NOT* considered.

* **Respect for Diversity**

  * Given the universality of black-box optimization (BBO) in science and engineering, different research communities have designed different methods and continue to increase. On the one hand, some of these methods may share *more or less* similarities. On the other hand, they may also show significant differences (w.r.t. motivations / objectives / implementations / practitioners). Therefore, we hope to cover such a diversity from different research communities such as artificial intelligence (particularly machine learning (**[evolutionary computation](https://github.com/Evolutionary-Intelligence/DistributedEvolutionaryComputation/blob/main/Summary/EvolutionaryComputation.md)** and zeroth-order optimization)), mathematical optimization/programming (particularly global optimization), operations research / management science, automatic control, open-source software, and perhaps others.

* **Respect for Originality**

  * *“It is both enjoyable and educational to hear the ideas directly from the creators”.* (From Hennessy, J.L. and Patterson, D.A., 2019. Computer architecture: A quantitative approach (Sixth Edition). Elsevier.)

  * For each optimizer considered here, we expect to give its original/representative reference (including its good implementations/improvements). If you find some important reference missed here, please do NOT hesitate to contact us (we will be happy to add it if necessary).

* **Respect for Repeatability**

  * For randomized search, properly controlling randomness is very crucial to repeat numerical experiments. Here we follow the *Random Sampling* suggestions from [NumPy](https://numpy.org/doc/stable/reference/random/). In other worlds, you must **explicitly** set the random seed for each optimizer.

## Reference

* [https://sites.google.com/view/benchmarking-network](https://sites.google.com/view/benchmarking-network)

  * [https://sites.google.com/view/benchmarking-network/home/activities/ppsn-2022-workshop](https://sites.google.com/view/benchmarking-network/home/activities/ppsn-2022-workshop)

  * Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T. and Brockhoff, D., 2021. [COCO: A platform for comparing continuous optimizers in a black-box setting](https://www.tandfonline.com/doi/full/10.1080/10556788.2020.1808977). Optimization Methods and Software, 36(1), pp.114-144.

  * Moré, J.J. and Wild, S.M., 2009. [Benchmarking derivative-free optimization algorithms](https://epubs.siam.org/doi/abs/10.1137/080724083). SIAM Journal on Optimization, 20(1), pp.172-191.

  * Whitley, D., Rana, S., Dzubera, J. and Mathias, K.E., 1996. [Evaluating evolutionary algorithms](https://www.sciencedirect.com/science/article/pii/0004370295001247). Artificial Intelligence, 85(1-2), pp.245-276.

* Hutter, F., Kotthoff, L. and Vanschoren, J., 2019. [Automated machine learning: Methods, systems, challenges](https://www.automl.org/wp-content/uploads/2019/05/AutoML_Book.pdf). Springer Nature.

* Berahas, A.S., Cao, L., Choromanski, K. and Scheinberg, K., 2022. [A theoretical and empirical comparison of gradient approximations in derivative-free optimization](https://link.springer.com/article/10.1007/s10208-021-09513-z). Foundations of Computational Mathematics, 22(2), pp.507-560.

* Kochenderfer, M.J. and Wheeler, T.A., 2019. [Algorithms for optimization](https://algorithmsbook.com/optimization/). MIT Press.

  * Larson, J., Menickelly, M. and Wild, S.M., 2019. [Derivative-free optimization methods](https://www.cambridge.org/core/journals/acta-numerica/article/abs/derivativefree-optimization-methods/84479E2B03A9BFFE0F9CD46CF9FCD289). Acta Numerica, 28, pp.287-404.

  * Fermi, E., 1952. [Numerical solution of a minimum problem](https://www.osti.gov/servlets/purl/4377177). Los Alamos Scientific Lab., Los Alamos, NM.

* Ollivier, Y., Arnold, L., Auger, A. and Hansen, N., 2017. [Information-geometric optimization algorithms: A unifying picture via invariance principles](https://www.jmlr.org/papers/v18/14-467.html). Journal of Machine Learning Research, 18(18), pp.1-65.

  * Hansel, K., Moos, J. and Derstroff, C., 2021. [Benchmarking the natural gradient in policy gradient methods and evolution strategies](https://link.springer.com/chapter/10.1007/978-3-030-41188-6_7). Reinforcement Learning Algorithms: Analysis and Applications, pp.69-84.

  * He, X., Zheng, Z. and Zhou, Y., 2021. [MMES: Mixture model-based evolution strategy for large-scale optimization](https://ieeexplore.ieee.org/abstract/document/9244595). IEEE Transactions on Evolutionary Computation, 25(2), pp.320-333.

  * Choromanski, K., Pacchiano, A., Parker-Holder, J. and Tang, Y., 2019. [From complexity to simplicity: Adaptive es-active subspaces for blackbox optimization](https://papers.nips.cc/paper/2019/hash/88bade49e98db8790df275fcebb37a13-Abstract.html). In Advances in Neural Information Processing Systems.

  * Liu, G., Zhao, L., Yang, F., Bian, J., Qin, T., Yu, N. and Liu, T.Y., 2019, July. [Trust region evolution strategies](https://ojs.aaai.org/index.php/AAAI/article/view/4345). In Proceedings of AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 4352-4359).

  * Loshchilov, I., Glasmachers, T. and Beyer, H.G., 2019. [Large scale black-box optimization by limited-memory matrix adaptation](https://ieeexplore.ieee.org/abstract/document/8410043). IEEE Transactions on Evolutionary Computation, 23(2), pp.353-358.

  * Varelas, K., Auger, A., Brockhoff, D., Hansen, N., ElHara, O.A., Semet, Y., Kassab, R. and Barbaresco, F., 2018, September. [A comparative study of large-scale variants of CMA-ES](https://link.springer.com/chapter/10.1007/978-3-319-99253-2_1). In International Conference on Parallel Problem Solving from Nature (pp. 3-15). Springer, Cham.

  * Müller, N. and Glasmachers, T., 2018, September. [Challenges in high-dimensional reinforcement learning with evolution strategies](https://link.springer.com/chapter/10.1007/978-3-319-99259-4_33). In International Conference on Parallel Problem Solving from Nature (pp. 411-423). Springer, Cham.

  * Li, Z. and Zhang, Q., 2018. [A simple yet efficient evolution strategy for large-scale black-box optimization](https://ieeexplore.ieee.org/abstract/document/8080257). IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.

  * Lehman, J., Chen, J., Clune, J. and Stanley, K.O., 2018, July. [ES is more than just a traditional finite-difference approximator](https://dl.acm.org/doi/abs/10.1145/3205455.3205474). In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 450-457). ACM.

  * Loshchilov, I., 2017. [LM-CMA: An alternative to L-BFGS for large-scale black box optimization](https://direct.mit.edu/evco/article-abstract/25/1/143/1041/LM-CMA-An-Alternative-to-L-BFGS-for-Large-Scale). Evolutionary Computation, 25(1), pp.143-171.

  * Krause, O., Arbonès, D.R. and Igel, C., 2016. [CMA-ES with optimal covariance update and storage complexity](https://proceedings.neurips.cc/paper/2016/hash/289dff07669d7a23de0ef88d2f7129e7-Abstract.html). In Advances in Neural Information Processing Systems, 29, pp.370-378.

  * Akimoto, Y. and Hansen, N., 2016, July. [Projection-based restricted covariance matrix adaptation for high dimension](https://dl.acm.org/doi/abs/10.1145/2908812.2908863). In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 197-204). ACM.

  * Hansen, N., Arnold, D.V. and Auger, A., 2015. [Evolution strategies](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44). In Springer Handbook of Computational Intelligence (pp. 871-898). Springer, Berlin, Heidelberg.

  * Hansen, N., Atamna, A. and Auger, A., 2014, September. [How to assess step-size adaptation mechanisms in randomised search](https://link.springer.com/chapter/10.1007/978-3-319-10762-2_6). In International Conference on Parallel Problem Solving from Nature (pp. 60-69). Springer, Cham.

  * Akimoto, Y., Auger, A. and Hansen, N., 2014, July. [Comparison-based natural gradient optimization in high dimension](https://dl.acm.org/doi/abs/10.1145/2576768.2598258). In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 373-380). ACM.

  * Hansen, N. and Auger, A., 2014. [Principled design of continuous stochastic search: From theory to practice](https://link.springer.com/chapter/10.1007/978-3-642-33206-7_8). In Theory and Principled Methods for the Design of Metaheuristics (pp. 145-180). Springer, Berlin, Heidelberg.
  
  * Bäck, T., Foussette, C. and Krause, P., 2013. [Contemporary evolution strategies](https://link.springer.com/book/10.1007/978-3-642-40137-4). Berlin: Springer.

  * Rudolph, G., 2012. [Evolutionary strategies](https://link.springer.com/referenceworkentry/10.1007/978-3-540-92910-9_22). In Handbook of Natural Computing (pp. 673-698). Springer Berlin, Heidelberg.

  * Akimoto, Y., Nagata, Y., Ono, I. and Kobayashi, S., 2012. [Theoretical foundation for CMA-ES from information geometry perspective](https://link.springer.com/article/10.1007/s00453-011-9564-8). Algorithmica, 64(4), pp.698-716.

  * Akimoto, Y., 2011. [Design of evolutionary computation for continuous optimization](https://drive.google.com/file/d/18PW9syYDy-ndJA7wBmE2hRlxXJRBTTir/view). Doctoral Dissertation, Tokyo Institute of Technology.

  * Suttorp, T., Hansen, N. and Igel, C., 2009. [Efficient covariance matrix update for variable metric evolution strategies](https://link.springer.com/article/10.1007/s10994-009-5102-1). Machine Learning, 75(2), pp.167-197.

  * Heidrich-Meisner, V. and Igel, C., 2008, September. [Evolution strategies for direct policy search](https://link.springer.com/chapter/10.1007/978-3-540-87700-4_43). In International Conference on Parallel Problem Solving from Nature (pp. 428-437). Springer, Berlin, Heidelberg.

  * Igel, C., Suttorp, T. and Hansen, N., 2006, July. [A computational efficient covariance matrix update and a (1+1)-CMA for evolution strategies](https://dl.acm.org/doi/abs/10.1145/1143997.1144082). In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 453-460). ACM.

  * Hansen, N., Müller, S.D. and Koumoutsakos, P., 2003. [Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES)](https://direct.mit.edu/evco/article-abstract/11/1/1/1139/Reducing-the-Time-Complexity-of-the-Derandomized). Evolutionary Computation, 11(1), pp.1-18.

  * Beyer, H.G. and Schwefel, H.P., 2002. [Evolution strategies–A comprehensive introduction](https://link.springer.com/article/10.1023/A:1015059928466). Natural Computing, 1(1), pp.3-52.

  * Hansen, N. and Ostermeier, A., 2001. [Completely derandomized self-adaptation in evolution strategies](https://direct.mit.edu/evco/article-abstract/9/2/159/892/Completely-Derandomized-Self-Adaptation-in). Evolutionary Computation, 9(2), pp.159-195.

  * Hansen, N. and Ostermeier, A., 1996, May. [Adapting arbitrary normal mutation distributions in evolution strategies: The covariance matrix adaptation](https://ieeexplore.ieee.org/abstract/document/542381). In Proceedings of IEEE International Conference on Evolutionary Computation (pp. 312-317). IEEE.

  * Rudolph, G., 1992. [On correlated mutations in evolution strategies](https://ls11-www.cs.tu-dortmund.de/people/rudolph/publications/papers/PPSN92.pdf). In International Conference on Parallel Problem Solving from Nature (pp. 105-114).

  * Rechenberg, I., 1989. [Evolution strategy: Nature’s way of optimization](https://link.springer.com/chapter/10.1007/978-3-642-83814-9_6). In Optimization: Methods and Applications, Possibilities and Limitations (pp. 106-126). Springer, Berlin, Heidelberg.

  * Schwefel, H.P., 1984. [Evolution strategies: A family of non-linear optimization techniques based on imitating some principles of organic evolution](https://link.springer.com/article/10.1007/BF01876146). Annals of Operations Research, 1(2), pp.165-167.

* Eiben, A.E. and Smith, J., 2015. [From evolutionary computation to the evolution of things](https://www.nature.com/articles/nature14544.). Nature, 521(7553), pp.476-482.

  * [http://www.evolutionarycomputation.org/](http://www.evolutionarycomputation.org/)

  * Beyer, H.G. and Deb, K., 2001. [On self-adaptive features in real-parameter evolutionary algorithms](https://ieeexplore.ieee.org/abstract/document/930314). IEEE Transactions on Evolutionary Computation, 5(3), pp.250-270.

  * Wolpert, D.H. and Macready, W.G., 1997. [No free lunch theorems for optimization](https://ieeexplore.ieee.org/document/585893). IEEE Transactions on Evolutionary Computation, 1(1), pp.67-82.

  * Bäck, T. and Schwefel, H.P., 1993. [An overview of evolutionary algorithms for parameter optimization](https://direct.mit.edu/evco/article-abstract/1/1/1/1092/An-Overview-of-Evolutionary-Algorithms-for). Evolutionary Computation, 1(1), pp.1-23.

* Schaul, T., Bayer, J., Wierstra, D., Sun, Y., Felder, M., Sehnke, F., Rückstieß, T. and Schmidhuber, J., 2010. [PyBrain](https://jmlr.org/papers/v11/schaul10a.html). Journal of Machine Learning Research, 11(24), pp.743-746.

  * Schaul, T., 2011. [Studies in continuous black-box optimization](https://people.idsia.ch/~schaul/publications/thesis.pdf). Doctoral Dissertation, Technische Universität München.

* Bonyadi, M.R. and Michalewicz, Z., 2017. [Particle swarm optimization for single objective continuous space problems: A review](https://direct.mit.edu/evco/article-abstract/25/1/1/1040/Particle-Swarm-Optimization-for-Single-Objective). Evolutionary Computation, 25(1), pp.1-54.

  * Poli, R., Kennedy, J. and Blackwell, T., 2007. [Particle swarm optimization](https://link.springer.com/article/10.1007/s11721-007-0002-0). Swarm Intelligence, 1(1), pp.33-57.

  * Eberhart, R.C., Shi, Y. and Kennedy, J., 2001. [Swarm intelligence](https://www.elsevier.com/books/swarm-intelligence/eberhart/978-1-55860-595-4). Elsevier.

## Research Support

This open-source Python library for black-box optimization is now supported by **Shenzhen Fundamental Research Program** under Grant No. JCYJ20200109141235597 (￥2,000,000), granted to **Prof. Yuhui Shi** (CSE, SUSTech @ Shenzhen, China), and actively developed (from 2021 to 2023) by his group members (e.g., *Qiqi Duan*, *Chang Shao*, *Guochen Zhou*, and Youkui Zhang).

We also acknowledge the initial discussions and Python code efforts (dating back to *about* 2017) with Hao Tong from the research group of **Prof. Yao** (CSE, SUSTech @ Shenzhen, China).
