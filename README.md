# pypop7 (Pure-PYthon library of POPulation-based black-box OPtimization)

[![GNU General Public License v3.0](https://img.shields.io/badge/license-GNU%20GPL--v3.0-green.svg)](https://github.com/Evolutionary-Intelligence/pypop/blob/main/LICENSE) [![gitter for pypop](https://img.shields.io/badge/gitter-pypop--go-brightgreen.svg)](https://gitter.im/pypop-go/community) [![PyPI for pypop7](https://img.shields.io/badge/PyPI-pypop7-yellowgreen.svg)](https://pypi.org/project/pypop7/) [![Documentation Status](https://readthedocs.org/projects/pypop/badge/?version=latest)](https://pypop.readthedocs.io/en/latest/?badge=latest) [![Downloads](https://pepy.tech/badge/pypop7)](https://pepy.tech/project/pypop7)

```PyPop7``` is a *Pure-PYthon* library of **POPulation-based OPtimization** for single-objective, real-parameter, black-box problems (**currently actively developed**). Its main goal is to provide a *unified* interface and *elegant* implementations for **Black-Box Optimization (BBO)**, *particularly population-based optimizers*, in order to facilitate research repeatability and also real-world applications.

<p align="center">
<img src="https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/logo/PyPop-Logo-Small-0.png" alt="drawing" width="321"/>
</p>

More specifically, for alleviating the notorious **curse of dimensionality** of BBO (almost based on *iterative sampling*), the primary focus of ```PyPop7``` is to cover their **State-Of-The-Art (SOTA) implementations for Large-Scale Optimization (LSO)**, though many of their other versions and variants are also included here (for benchmarking/mixing purpose, and sometimes even for practical purpose).

## How to Use PyPop7

The following three simple steps are enough to utilize the optimization power of [PyPop7](https://pypi.org/project/pypop7/):

1. Use [pip](https://pypi.org/project/pip/) to install ```pypop7``` on the Python3-based virtual environment via [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```bash
$ pip install pypop7
```

For simplicity, all required dependencies are *automatically* installed according to [setup.cfg](https://github.com/Evolutionary-Intelligence/pypop/blob/main/setup.cfg).

2. Define your own objective function for the optimization problem at hand,

3. Run one or more black-box optimizers from ```pypop7``` on the given optimization problem:

```Python
import numpy as np  # for numerical computation, which is also the computing engine of pypop7

# 2. Define your own objective function for the optimization problem at hand:
#   the below example is Rosenbrock, the notorious test function in the optimization community
def rosenbrock(x):
    return 100 * np.sum(np.power(x[1:] - np.power(x[:-1], 2), 2)) + np.sum(np.power(x[:-1] - 1, 2))

# define the fitness (cost) function and also its settings
ndim_problem = 1000
problem = {'fitness_function': rosenbrock,  # cost function
           'ndim_problem': ndim_problem,  # dimension
           'lower_boundary': -5 * np.ones((ndim_problem,)),  # search boundary
           'upper_boundary': 5 * np.ones((ndim_problem,))}

# 3. Run one or more black-box optimizers from ```pypop7``` on the given optimization problem:
#   here we choose LM-MA-ES owing to its low complexity and metric-learning ability for LSO
from pypop7.optimizers.es.lmmaes import LMMAES
# define all the necessary algorithm options (which differ among different optimizers)
options = {'fitness_threshold': 1e-10,  # terminate when the best-so-far fitness is lower than this threshold
           'max_runtime': 3600,  # 1 hours (terminate when the actual runtime exceeds it)
           'seed_rng': 0,  # seed of random number generation (which must be explicitly set for repeatability)
           'x': 4 * np.ones((ndim_problem,)),  # initial mean of search (mutation/sampling) distribution
           'sigma': 0.3,  # initial global step-size of search distribution
           'verbose_frequency': 500}
lmmaes = LMMAES(problem, options)  # initialize the optimizer
results = lmmaes.optimize()  # run its (time-consuming) search process
print(results)
```

Below [DEMOs](https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/demo/) are given on a toy 2-dimensional minimization function, in order to visually show the very interesting/powerful evolutionary search process of [```MAES```](https://ieeexplore.ieee.org/abstract/document/7875115/) and [```LMCMAES```](https://dl.acm.org/doi/abs/10.1145/2576768.2598294):

| MA-ES | LM-CMA-ES |
| :---: | :-------: |
| <img src="https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/demo/demo_maes.gif" alt="drawing" width="200"/> | <img src="https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/demo/demo_lmcmaes.gif" alt="drawing" width="200"/> |
| **Hooke-Jeeves** (1961) | **Nelder-Mead** (1965) |
| <img src="https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/demo/demo_hj.gif" alt="drawing" width="200"/> | <img src="https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/demo/demo_nm.gif" alt="drawing" width="200"/> |

## A (*Still Growing*) List of Publicly Available Black-Box Optimizers (BBO)

******* *** *******

![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg): indicates the specific version for LSO (e.g., dimension >= 1000).

![competitor](https://img.shields.io/badge/**-competitor-blue.svg): indicates the competitive (or *de facto*) version for *relatively low-dimensional* problems (though it may also work well under certain LSO circumstances).

![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg): indicates the baseline version for benchmarking purpose or for theoretical interest.

******* *** *******

* **Evolution Strategies (ES)** [See e.g. [Ollivier et al., 2017, JMLR](https://www.jmlr.org/papers/v18/14-467.html); [Hansen et al., 2015](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44); [Bäck et al., 2013](https://link.springer.com/book/10.1007/978-3-642-40137-4); [Rudolph, 2012](https://link.springer.com/referenceworkentry/10.1007/978-3-540-92910-9_22); [Beyer&Schwefel, 2002](https://link.springer.com/article/10.1023/A:1015059928466); [Rechenberg, 1989](https://link.springer.com/chapter/10.1007/978-3-642-83814-9_6); [Schwefel, 1984](https://link.springer.com/article/10.1007/BF01876146)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Mixture Model-based Evolution Strategy (**[MMES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/mmes.py)**) [See [He et al., 2021, TEVC](https://ieeexplore.ieee.org/abstract/document/9244595)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Limited-Memory Matrix Adaptation Evolution Strategy (**[LMMAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/lmmaes.py)**) [see [Loshchilov et al., 2019, TEVC](https://ieeexplore.ieee.org/abstract/document/8410043)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Limited Memory Covariance Matrix Adaptation (**[LMCMA](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/lmcma.py)**) [see [Loshchilov, 2017, ECJ](https://direct.mit.edu/evco/article-abstract/25/1/143/1041/LM-CMA-An-Alternative-to-L-BFGS-for-Large-Scale)]
  
    * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Limited Memory Covariance Matrix Adaptation Evolution Strategy (**[LMCMAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/lmcmaes.py)**) [see [Loshchilov, 2014, GECCO](https://dl.acm.org/doi/abs/10.1145/2576768.2598294)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Rank-m Evolution Strategy *with Multiple Evolution Paths* (**[RMES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/rmes2.py)**, Rm-ES) [See [Li&Zhang, 2018, TEVC](https://ieeexplore.ieee.org/document/8080257)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Rank-One Evolution Strategy (**[R1ES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/r1es.py)**) [see [Li&Zhang, 2018, TEVC](https://ieeexplore.ieee.org/document/8080257)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Projection-based Covariance Matrix Adaptation (**VKDCMA**, VkD-CMA) [See [Akimoto&Hansen, 2016, GECCO](https://dl.acm.org/doi/abs/10.1145/2908812.2908863)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Linear Covariance Matrix Adaptation (**[VDCMA](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/vdcma.py)**, VD-CMA) [See [Akimoto et al., 2014, GECCO](https://dl.acm.org/doi/abs/10.1145/2576768.2598258)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Cholesky-CMA-ES-2016 (**[CCMAES2016](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/ccmaes2016.py)**) [See [Krause et al., 2016, NeurIPS](https://proceedings.neurips.cc/paper/2016/hash/289dff07669d7a23de0ef88d2f7129e7-Abstract.html)]

    * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) (1+1)-Active-Cholesky-CMA-ES-2015 (**[OPOA2015](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/opoa2015.py)**) [See [Krause&Igel, 2015, FOGA](https://dl.acm.org/doi/abs/10.1145/2725494.2725496)]

    * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) (1+1)-Active-Cholesky-CMA-ES (**[OPOA](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/opoa.py)**) [See [Arnold&Hansen, 2010, GECCO](https://dl.acm.org/doi/abs/10.1145/1830483.1830556)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Cholesky-CMA-ES (**[CCMAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/ccmaes.py)**) [See [Suttorp et al., 2009, MLJ](https://link.springer.com/article/10.1007/s10994-009-5102-1)]
 
    * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) (1+1)-Cholesky-CMA-ES-2009 (**[OPOC2009](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/opoc2009.py)**) [See [Suttorp et al., 2009, MLJ](https://link.springer.com/article/10.1007/s10994-009-5102-1)]

    * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) (1+1)-Cholesky-CMA-ES (**[OPOC](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/opoc.py)**) [See [Igel et al., 2006, GECCO](https://dl.acm.org/doi/abs/10.1145/1143997.1144082)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Separable Covariance Matrix Adaptation Evolution Strategy (**[SEPCMAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/sepcmaes.py)**, sep-CMA-ES) [See [Bäck et al., 2013](https://link.springer.com/book/10.1007/978-3-642-40137-4); [Ros&Hansen, 2008, PPSN](https://link.springer.com/chapter/10.1007/978-3-540-87700-4_30)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Main Vector Adaptation Evolution Strategies (**[MVAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/mvaes.py)**, MVA-ES) [See [Poland&Zell, 2001, GECCO](https://dl.acm.org/doi/abs/10.5555/2955239.2955428)]

  * ![competitor](https://img.shields.io/badge/**-competitor-blue.svg) Diagonal Decoding Covariance Matrix Adaptation (**DDCMA**, dd-CMA) [See [Akimoto&Hansen, 2019, ECJ](https://direct.mit.edu/evco/article/28/3/405/94999/Diagonal-Acceleration-for-Covariance-Matrix)]

  * ![competitor](https://img.shields.io/badge/**-competitor-blue.svg) Covariance Matrix Self-Adaptation with Repelling Subpopulations (**RSCMSA**, RS-CMSA) [See [Ahrari et al., 2017, ECJ](https://doi.org/10.1162/evco_a_00182)]

  * ![competitor](https://img.shields.io/badge/**-competitor-blue.svg) Matrix Adaptation Evolution Strategy (**[MAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/maes.py)**) [see [Beyer&Sendhoff, 2017, TEVC](https://ieeexplore.ieee.org/abstract/document/7875115/)]

    * ![competitor](https://img.shields.io/badge/**-competitor-blue.svg) Fast Matrix Adaptation Evolution Strategy (**[FMAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/fmaes.py)**) [see [Beyer, 2020, GECCO](https://dl.acm.org/doi/abs/10.1145/3377929.3389870); [Loshchilov et al., 2019, TEVC](https://ieeexplore.ieee.org/abstract/document/8410043)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Self-Adaptation Evolution Strategy (**[SAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/saes.py)**) [see e.g. [Beyer, 2020, GECCO](https://dl.acm.org/doi/abs/10.1145/3377929.3389870); [Beyer, 2007, Scholarpedia](http://www.scholarpedia.org/article/Evolution_strategies)]

    * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Cumulative Step-size Adaptation Evolution Strategy (**[CSAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/csaes.py)**)  [see e.g. [Hansen et al., 2015](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44); [Ostermeier et al., 1994, PPSN](https://link.springer.com/chapter/10.1007/3-540-58484-6_263)]

    * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Derandomized Self-Adaptation Evolution Strategy (**[DSAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/dsaes.py)**) [see e.g. [Hansen et al., 2015](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44); [Ostermeier et al., 1994, ECJ](https://direct.mit.edu/evco/article-abstract/2/4/369/1407/A-Derandomized-Approach-to-Self-Adaptation-of)]

    * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Schwefel's Self-Adaptation Evolution Strategy (**[SSAES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/ssaes.py)**) [see e.g. [Hansen et al., 2015](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44)]

    * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Rechenberg's (1+1)-Evolution Strategy with 1/5th success rule (**[RES](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/res.py)**) [see e.g. [Hansen et al., 2015](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44); [Kern et al., 2004](https://link.springer.com/article/10.1023/B:NACO.0000023416.59689.4e); [Schumer&Steiglitz, 1968, IEEE-TAC](https://ieeexplore.ieee.org/abstract/document/1098903)]

* **Natural Evolution Strategies (NES)** [See e.g. [Wierstra et al., 2014, JMLR](https://jmlr.org/papers/v15/wierstra14a.html); [Yi et al., 2009, ICML](https://dl.acm.org/doi/abs/10.1145/1553374.1553522); [Wierstra et al., 2008, CEC](https://ieeexplore.ieee.org/document/4631255)]

  * Rank-One Natural Evolution Strategy (**R1NES**) [See [Sun et al., 2013, GECCO](https://dl.acm.org/doi/abs/10.1145/2464576.2464608)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Separable Natural Evolution Strategy (**SNES**) [See [Schaul et al., 2011, GECCO](https://dl.acm.org/doi/abs/10.1145/2001576.2001692)]

* **Estimation of Distribution Algorithms** (**[EDA](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/eda/eda.py)**) [See e.g. [Larrañaga&Lozano, 2002](https://link.springer.com/book/10.1007/978-1-4615-1539-5); [Pelikan et al., 2002](https://link.springer.com/article/10.1023/A:1013500812258); [Mühlenbein&Paaß, 1996, PPSN](https://link.springer.com/chapter/10.1007/3-540-61723-X_982)]

  * ![large--scale--optimization](https://img.shields.io/badge/***-large--scale--optimization-orange.svg) Univariate Marginal Distribution Algorithm (**[UMDA](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/eda/umda.py)**) [see e.g. [Larrañaga&Lozano, 2001](https://link.springer.com/book/10.1007/978-1-4615-1539-5); [Mühlenbein, 1997, ECJ](https://tinyurl.com/yt78c786)]

* **Cross-Entropy Method** (**CEM**) [See e.g. [Rubinstein&Kroese, 2004](https://link.springer.com/book/10.1007/978-1-4757-4321-0)]

* **Differential Evolution ([DE](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/de/de.py))** [see e.g. [Storn&Price, 1997, JGO](https://link.springer.com/article/10.1023/A:1008202821328)]

  * ![competitor](https://img.shields.io/badge/**-competitor-blue.svg) Composite Differential Evolution (**[CODE](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/de/code.py)**) [see [Wang et al., 2011, TEVC](https://doi.org/10.1109/TEVC.2010.2087271)]

  * ![competitor](https://img.shields.io/badge/**-competitor-blue.svg) Adaptive Differential Evolution (**[JADE](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/de/jade.py)**) [see [Zhang&Sanderson, 2009, TEVC](https://doi.org/10.1109/TEVC.2009.2014613)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Trigonometric-mutation Differential Evolution (**[TDE](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/de/tde.py)**) [see e.g. [Fan&Lampinen, 2003, JGO](https://link.springer.com/article/10.1023/A:1024653025686)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Classic Differential Evolution (**[CDE](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/de/cde.py)**) [see e.g. [Storn&Price, 1997, JGO](https://link.springer.com/article/10.1023/A:1008202821328)]

* **Particle Swarm Optimization (PSO)** [See e.g. [Shi&Eberhart, 1998, CEC](https://ieeexplore.ieee.org/abstract/document/699146); [Kennedy&Eberhart, 1995, ICNN](https://ieeexplore.ieee.org/document/488968)]

* **CoOperative co-Evolutionary Algorithms (COEA)** [See e.g. [Gomez et al., 2008, JMLR](https://jmlr.org/papers/v9/gomez08a.html); [Panait et al., 2008, JMLR](https://www.jmlr.org/papers/v9/panait08a.html)]

  * CoOperative SYnapse NeuroEvolution (**COSYNE**, CoSyNE) [See [Gomez et al., 2008, JMLR](https://jmlr.org/papers/v9/gomez08a.html)]

* **Simulated Annealing (SA)** [See e.g. [Kirkpatrick et al., 1983, Science](https://www.science.org/doi/10.1126/science.220.4598.671); [Hastings, 1970, Biometrika](https://academic.oup.com/biomet/article/57/1/97/284580); [Metropolis et al., 1953, JCP](https://aip.scitation.org/doi/abs/10.1063/1.1699114)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Enhanced Simulated Annealing (**ESA**) [See [Siarry et al., 1997, ACM-TOMS](https://dl.acm.org/doi/abs/10.1145/264029.264043)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Corana et al.' Simulated Annealing (**CSA**) [See [Corana et al., 1987, ACM-TOMS](https://dl.acm.org/doi/abs/10.1145/29380.29864)]

* **Genetic Algorithms (GA)** [see e.g. [Forrest, 1993, Science](https://www.science.org/doi/abs/10.1126/science.8346439); [Holland, 1962, JACM](https://dl.acm.org/doi/10.1145/321127.321128)]

* **Evolutionary Programming ([EP](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ep/ep.py))** [see e.g. [Yao et al., 1999, TEVC](https://ieeexplore.ieee.org/abstract/document/771163)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Fast Evolutionary Programming (**[FEP](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ep/fep.py)**) [see [Yao et al., 1999, TEVC](https://ieeexplore.ieee.org/abstract/document/771163)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Classical Evolutionary Programming (**[CEP](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ep/cep.py)**) [see e.g. [Yao et al., 1999, TEVC](https://ieeexplore.ieee.org/abstract/document/771163); [Bäck&Schwefel, 1993, ECJ](https://direct.mit.edu/evco/article-abstract/1/1/1/1092/An-Overview-of-Evolutionary-Algorithms-for)]

* **Direct Search ([DS](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ds/ds.py))** [see e.g. [Powell, 1998, Acta-Numerica](https://www.cambridge.org/core/journals/acta-numerica/article/abs/direct-search-algorithms-for-optimization-calculations/23FA5B19EAF122E02D3724DB1841238C); [Wright, 1996](https://nyuscholars.nyu.edu/en/publications/direct-search-methods-once-scorned-now-respectable); [Hooke&Jeeves, 1961, JACM](https://dl.acm.org/doi/10.1145/321062.321069)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Generalized Pattern Search (**[GPS](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ds/gps.py)**) [see [Kochenderfer&Wheeler, 2019](https://algorithmsbook.com/optimization/files/chapter-7.pdf); [Torczon, 1997, SIAM-JO](https://epubs.siam.org/doi/abs/10.1137/S1052623493250780)]
  
  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Nelder-Mead simplex method (**[NM](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ds/nm.py)**) [see [Nelder&Mead, 1965, Computer](https://academic.oup.com/comjnl/article-abstract/7/4/308/354237)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Hooke-Jeeves direct search method (**[HJ](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ds/hj.py)**) [see [Kochenderfer&Wheeler, 2019](https://algorithmsbook.com/optimization/files/chapter-7.pdf); [Kaupe, 1963, CACM](https://dl.acm.org/doi/pdf/10.1145/366604.366632); [Hooke&Jeeves, 1961, JACM](https://dl.acm.org/doi/10.1145/321062.321069)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Coordinate Search (**[CS](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/ds/cs.py)**) [see [Torczon, 1997, SIAM-JO](https://epubs.siam.org/doi/abs/10.1137/S1052623493250780); [Fermi&Metropolis, 1952](https://www.osti.gov/servlets/purl/4377177)]

* **Random (stochastic) Search ([RS](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/rs.py))** [see e.g. [Bergstra&Bengio, 2012, JMLR](https://www.jmlr.org/papers/v13/bergstra12a.html); [Rastrigin, 1986](https://link.springer.com/content/pdf/10.1007/BFb0007129.pdf); [Solis&Wets, 1981, MOOR](https://pubsonline.informs.org/doi/abs/10.1287/moor.6.1.19); [Brooks, 1958, OR](https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Simple Random Search (**[SRS](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/srs.py)**) [see [Rosenstein&Barto, 2001, IJCAI](https://dl.acm.org/doi/abs/10.5555/1642194.1642206)]
  
  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Annealed Random Hill Climber (**[ARHC](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/arhc.py)**) [see e.g. [Schaul et al., 2010, JMLR](https://jmlr.org/papers/v11/schaul10a.html)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Random Hill Climber (**[RHC](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/rhc.py)**) [see e.g. [Schaul et al., 2010, JMLR](https://jmlr.org/papers/v11/schaul10a.html)]

  * ![baseline](https://img.shields.io/badge/*-baseline-lightgrey.svg) Pure Random Search (**[PRS](https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/prs.py)**) [see e.g. [Bergstra&Bengio, 2012, JMLR](https://www.jmlr.org/papers/v13/bergstra12a.html)]

## Design Philosophy

* **Respect for Beauty (Elegance)**

  * From the *problem-solving* perspective, we empirically prefer to choose the *best* optimizer for the black-box optimization problem at hand. However, for the *new* problem, the *best* optimizer is often unknown in advance (without *a prior* knowledge). As a rule of thumb, we need to compare a (often small) set of all available/well-known optimizers and choose the *best* one from them according to some predefined performance criteria. From the *research* perspective, however, we like *beautiful* optimizers, though always keeping the **[“No Free Lunch” theorem](https://ieeexplore.ieee.org/document/585893)** in mind. Typically, the **beauty** of one optimizer comes from the following features: **novelty** (e.g., GA/PSO), **competitive performance** on at least one class of problems (e.g., BO), **theoretical insights** (e.g., CMA-ES/NES), **clarity/simplicity** (e.g., CEM/EDA), and **repeatability**.

    * "If there is a single dominant theme in this ..., it is that practical methods of numerical computation can be simultaneously efficient, clever, and — important — clear." (From Press, W.H., Teukolsky, S.A., Vetterling, W.T. and Flannery, B.P., 2007. [Numerical recipes: The art of scientific computing](http://numerical.recipes/). Cambridge University Press.)

  * If you find any BBO/DFO to meet the above standard, welcome to launch [issues](https://github.com/Evolutionary-Intelligence/pypop/issues) or [pulls](https://github.com/Evolutionary-Intelligence/pypop/pulls). We will consider it to be included in the ```pypop``` library. Note that **any [superficial](https://onlinelibrary.wiley.com/doi/full/10.1111/itor.13176) [imitation](https://dl.acm.org/doi/10.1145/3402220.3402221)** to the above well-established optimizers (**['Old Wine in a New Bottle'](https://link.springer.com/article/10.1007/s11721-021-00202-9)**) will be *NOT* considered.

* **Respect for Diversity**

  * Given the universality of black-box optimization (BBO) in science and engineering, different research communities have designed different methods and continue to increase. On the one hand, some of these methods may share *more or less* similarities. On the other hand, they may also show significant differences (w.r.t. motivations / objectives / implementations / practitioners). Therefore, we hope to cover such a diversity from different research communities such as artificial intelligence (particularly machine learning (**[evolutionary computation](https://github.com/Evolutionary-Intelligence/DistributedEvolutionaryComputation/blob/main/Summary/EvolutionaryComputation.md)** and zeroth-order optimization)), mathematical optimization/programming (particularly global optimization), operations research / management science, automatic control, open-source software, and perhaps others.

* **Respect for Originality**

  * *“It is both enjoyable and educational to hear the ideas directly from the creators”.* (From Hennessy, J.L. and Patterson, D.A., 2019. Computer architecture: A quantitative approach (Sixth Edition). Elsevier.)

  * For each optimizer considered here, we expect to give its original/representative reference (including its good implementations/improvements). If you find some important reference missed here, please do NOT hesitate to contact us (we will be happy to add it if necessary).

* **Respect for Repeatability**

  * For randomized search, properly controlling randomness is very crucial to repeat numerical experiments. Here we follow the *Random Sampling* suggestions from [NumPy](https://numpy.org/doc/stable/reference/random/). In other worlds, you must **explicitly** set the random seed for each optimizer.

## Computational Efficiency

For LSO, computational efficiency is an indispensable performance criterion of DFO [in the post-Moore era](https://www.science.org/doi/10.1126/science.aam9744). To obtain high-performance computation as much as possible, [NumPy](https://www.nature.com/articles/s41586-020-2649-2) is heavily used in this library as the base of numerical computation along with [SciPy](https://www.nature.com/articles/s41592-019-0686-2). Sometimes, [Numba](https://numba.pydata.org/) is also utilized, in order to further accelerate the wall-clock time.

## Development Guide

[PEP 257 – Docstring Conventions](https://peps.python.org/pep-0257/)

Since this library is built on the wonderful numerical computing library NumPy, we further use the Docstring Conventions from NumPy: [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html).

## Reference

* [https://sites.google.com/view/benchmarking-network](https://sites.google.com/view/benchmarking-network) | [PPSN-2022](https://sites.google.com/view/benchmarking-network/home/activities/ppsn-2022-workshop)

  * Meunier, L., Rakotoarison, H., Wong, P.K., Roziere, B., Rapin, J., Teytaud, O., Moreau, A. and Doerr, C., 2022. [Black-box optimization revisited: Improving algorithm selection wizards through massive benchmarking](https://ieeexplore.ieee.org/abstract/document/9524335). IEEE Transactions on Evolutionary Computation, 26(3), pp.490-500.

  * Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T. and Brockhoff, D., 2021. [COCO: A platform for comparing continuous optimizers in a black-box setting](https://www.tandfonline.com/doi/full/10.1080/10556788.2020.1808977). Optimization Methods and Software, 36(1), pp.114-144.

  * Auger, A. and Hansen, N., 2021, July. [Benchmarking: State-of-the-art and beyond](https://dl.acm.org/doi/abs/10.1145/3449726.3461424). In Proceedings of Genetic and Evolutionary Computation Conference Companion (pp. 339-340). ACM.

  * Varelas, K., El Hara, O.A., Brockhoff, D., Hansen, N., Nguyen, D.M., Tušar, T. and Auger, A., 2020. [Benchmarking large-scale continuous optimizers: The bbob-largescale testbed, a COCO software guide and beyond](https://www.sciencedirect.com/science/article/abs/pii/S156849462030675X). Applied Soft Computing, 97, p.106737.

  * Moré, J.J. and Wild, S.M., 2009. [Benchmarking derivative-free optimization algorithms](https://epubs.siam.org/doi/abs/10.1137/080724083). SIAM Journal on Optimization, 20(1), pp.172-191.

  * Whitley, D., Rana, S., Dzubera, J. and Mathias, K.E., 1996. [Evaluating evolutionary algorithms](https://www.sciencedirect.com/science/article/pii/0004370295001247). Artificial Intelligence, 85(1-2), pp.245-276.
  
  * Salomon, R., 1996. [Re-evaluating genetic algorithm performance under coordinate rotation of benchmark functions. A survey of some theoretical and practical aspects of genetic algorithms](https://www.sciencedirect.com/science/article/abs/pii/0303264796016218). BioSystems, 39(3), pp.263-278.

  * Moré, J.J., Garbow, B.S. and Hillstrom, K.E., 1981. [Testing unconstrained optimization software](https://dl.acm.org/doi/10.1145/355934.355936). ACM Transactions on Mathematical Software, 7(1), pp.17-41.

* Hutter, F., Kotthoff, L. and Vanschoren, J., 2019. [Automated machine learning: Methods, systems, challenges](https://www.automl.org/wp-content/uploads/2019/05/AutoML_Book.pdf). Springer Nature.

  * Hoos, H.H., 2011. [Automated algorithm configuration and parameter tuning](https://link.springer.com/chapter/10.1007/978-3-642-21434-9_3). In Autonomous Search (pp. 37-71). Springer, Berlin, Heidelberg.

* Berahas, A.S., Cao, L., Choromanski, K. and Scheinberg, K., 2022. [A theoretical and empirical comparison of gradient approximations in derivative-free optimization](https://link.springer.com/article/10.1007/s10208-021-09513-z). Foundations of Computational Mathematics, 22(2), pp.507-560.

  * Porcelli, M. and Toint, P.L., 2022. [Exploiting problem structure in derivative free optimization](https://dl.acm.org/doi/abs/10.1145/3474054). ACM Transactions on Mathematical Software, 48(1), pp.1-25.

  * Gao, K. and Sener, O., 2022, June. [Generalizing Gaussian smoothing for random search](https://proceedings.mlr.press/v162/gao22f.html). In International Conference on Machine Learning (pp. 7077-7101). PMLR.

  * Kochenderfer, M.J. and Wheeler, T.A., 2019. [Algorithms for optimization](https://algorithmsbook.com/optimization/). MIT Press.

  * Larson, J., Menickelly, M. and Wild, S.M., 2019. [Derivative-free optimization methods](https://www.cambridge.org/core/journals/acta-numerica/article/abs/derivativefree-optimization-methods/84479E2B03A9BFFE0F9CD46CF9FCD289). Acta Numerica, 28, pp.287-404.

  * Nesterov, Y., 2018. [Lectures on convex optimization](https://link.springer.com/book/10.1007/978-3-319-91578-4). Berlin: Springer International Publishing.

  * Nesterov, Y. and Spokoiny, V., 2017. [Random gradient-free minimization of convex functions](https://link.springer.com/article/10.1007/s10208-015-9296-2). Foundations of Computational Mathematics, 17(2), pp.527-566.

  * Conn, A.R., Scheinberg, K. and Vicente, L.N., 2009. [Introduction to derivative-free optimization](https://epubs.siam.org/doi/book/10.1137/1.9780898718768). Society for Industrial and Applied Mathematics.

* Ollivier, Y., Arnold, L., Auger, A. and Hansen, N., 2017. [Information-geometric optimization algorithms: A unifying picture via invariance principles](https://www.jmlr.org/papers/v18/14-467.html). Journal of Machine Learning Research, 18(18), pp.1-65. [ [A visual guide to evolution strategies](https://blog.otoro.net/2017/10/29/visual-evolution-strategies/) ]

  * Glasmachers, T. and Krause, O., 2022. [Convergence analysis of the Hessian estimation evolution strategy](https://direct.mit.edu/evco/article-abstract/30/1/27/102711/Convergence-Analysis-of-the-Hessian-Estimation). Evolutionary Computation, 30(1), pp.27-50.

  * Akimoto, Y. and Hansen, N., 2022, July. [CMA-ES and advanced adaptation mechanisms](http://www.cmap.polytechnique.fr/~nikolaus.hansen/gecco-2022-cma-tutorial.pdf). In Proceedings of Annual Conference on Genetic and Evolutionary Computation Companion. ACM.

  * Hansel, K., Moos, J. and Derstroff, C., 2021. [Benchmarking the natural gradient in policy gradient methods and evolution strategies](https://link.springer.com/chapter/10.1007/978-3-030-41188-6_7). Reinforcement Learning Algorithms: Analysis and Applications, pp.69-84.

  * He, X., Zheng, Z. and Zhou, Y., 2021. [MMES: Mixture model-based evolution strategy for large-scale optimization](https://ieeexplore.ieee.org/abstract/document/9244595). IEEE Transactions on Evolutionary Computation, 25(2), pp.320-333.

  * Li, Z., Lin, X., Zhang, Q. and Liu, H., 2020. [Evolution strategies for continuous optimization: A survey of the state-of-the-art](https://www.sciencedirect.com/science/article/abs/pii/S221065021930584X). Swarm and Evolutionary Computation, 56, p.100694.

  * Akimoto, Y. and Hansen, N., 2020. [Diagonal acceleration for covariance matrix adaptation evolution strategies](https://direct.mit.edu/evco/article/28/3/405/94999/Diagonal-Acceleration-for-Covariance-Matrix). Evolutionary Computation, 28(3), pp.405-435.

  * Choromanski, K., Pacchiano, A., Parker-Holder, J. and Tang, Y., 2019. [From complexity to simplicity: Adaptive es-active subspaces for blackbox optimization](https://papers.nips.cc/paper/2019/hash/88bade49e98db8790df275fcebb37a13-Abstract.html). In Advances in Neural Information Processing Systems.

  * Loshchilov, I., Glasmachers, T. and Beyer, H.G., 2019. [Large scale black-box optimization by limited-memory matrix adaptation](https://ieeexplore.ieee.org/abstract/document/8410043). IEEE Transactions on Evolutionary Computation, 23(2), pp.353-358.

  * Li, Z., Zhang, Q., Lin, X. and Zhen, H.L., 2018. [Fast covariance matrix adaptation for large-scale black-box optimization](https://ieeexplore.ieee.org/abstract/document/8533604). IEEE Transactions on Cybernetics, 50(5), pp.2073-2083.

  * Varelas, K., Auger, A., Brockhoff, D., Hansen, N., ElHara, O.A., Semet, Y., Kassab, R. and Barbaresco, F., 2018, September. [A comparative study of large-scale variants of CMA-ES](https://link.springer.com/chapter/10.1007/978-3-319-99253-2_1). In International Conference on Parallel Problem Solving from Nature (pp. 3-15). Springer, Cham.

  * Li, Z. and Zhang, Q., 2018. [A simple yet efficient evolution strategy for large-scale black-box optimization](https://ieeexplore.ieee.org/abstract/document/8080257). IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.

  * Lehman, J., Chen, J., Clune, J. and Stanley, K.O., 2018, July. [ES is more than just a traditional finite-difference approximator](https://dl.acm.org/doi/abs/10.1145/3205455.3205474). In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 450-457). ACM.

  * Loshchilov, I., 2017. [LM-CMA: An alternative to L-BFGS for large-scale black box optimization](https://direct.mit.edu/evco/article-abstract/25/1/143/1041/LM-CMA-An-Alternative-to-L-BFGS-for-Large-Scale). Evolutionary Computation, 25(1), pp.143-171.

  * Krause, O., Arbonès, D.R. and Igel, C., 2016. [CMA-ES with optimal covariance update and storage complexity](https://proceedings.neurips.cc/paper/2016/hash/289dff07669d7a23de0ef88d2f7129e7-Abstract.html). In Advances in Neural Information Processing Systems, 29, pp.370-378.

  * Akimoto, Y. and Hansen, N., 2016, July. [Projection-based restricted covariance matrix adaptation for high dimension](https://dl.acm.org/doi/abs/10.1145/2908812.2908863). In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 197-204). ACM.

  * Krause, O. and Igel, C., 2015, January. [A more efficient rank-one covariance matrix update for evolution strategies](https://dl.acm.org/doi/abs/10.1145/2725494.2725496). In Proceedings of ACM Conference on Foundations of Genetic Algorithms (pp. 129-136). ACM.

  * Hansen, N., Arnold, D.V. and Auger, A., 2015. [Evolution strategies](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_44). In Springer Handbook of Computational Intelligence (pp. 871-898). Springer, Berlin, Heidelberg.

  * Loshchilov, I., 2014, July. [A computationally efficient limited memory CMA-ES for large scale optimization](https://dl.acm.org/doi/abs/10.1145/2576768.2598294). In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 397-404). ACM.

  * Hansen, N., Atamna, A. and Auger, A., 2014, September. [How to assess step-size adaptation mechanisms in randomised search](https://link.springer.com/chapter/10.1007/978-3-319-10762-2_6). In International Conference on Parallel Problem Solving from Nature (pp. 60-69). Springer, Cham.

  * Akimoto, Y., Auger, A. and Hansen, N., 2014, July. [Comparison-based natural gradient optimization in high dimension](https://dl.acm.org/doi/abs/10.1145/2576768.2598258). In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 373-380). ACM.

  * Hansen, N. and Auger, A., 2014. [Principled design of continuous stochastic search: From theory to practice](https://link.springer.com/chapter/10.1007/978-3-642-33206-7_8). In Theory and Principled Methods for the Design of Metaheuristics (pp. 145-180). Springer, Berlin, Heidelberg.
  
  * Bäck, T., Foussette, C. and Krause, P., 2013. [Contemporary evolution strategies](https://link.springer.com/book/10.1007/978-3-642-40137-4). Berlin: Springer.

  * Rudolph, G., 2012. [Evolutionary strategies](https://link.springer.com/referenceworkentry/10.1007/978-3-540-92910-9_22). In Handbook of Natural Computing (pp. 673-698). Springer Berlin, Heidelberg.

  * Akimoto, Y., Nagata, Y., Ono, I. and Kobayashi, S., 2012. [Theoretical foundation for CMA-ES from information geometry perspective](https://link.springer.com/article/10.1007/s00453-011-9564-8). Algorithmica, 64(4), pp.698-716.

  * Akimoto, Y., 2011. [Design of evolutionary computation for continuous optimization](https://drive.google.com/file/d/18PW9syYDy-ndJA7wBmE2hRlxXJRBTTir/view). Doctoral Dissertation, Tokyo Institute of Technology.
  
  * Akimoto, Y., Nagata, Y., Ono, I. and Kobayashi, S., 2010, September. [Bidirectional relation between CMA evolution strategies and natural evolution strategies](https://link.springer.com/chapter/10.1007/978-3-642-15844-5_16). In International Conference on Parallel Problem Solving from Nature (pp. 154-163). Springer, Berlin, Heidelberg.

  * Arnold, D.V. and Hansen, N., 2010, July. [Active covariance matrix adaptation for the (1+1)-CMA-ES](https://dl.acm.org/doi/abs/10.1145/1830483.1830556). In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 385-392). ACM.

  * Heidrich-Meisner, V. and Igel, C., 2009, June. [Hoeffding and Bernstein races for selecting policies in evolutionary direct policy search](https://dl.acm.org/doi/abs/10.1145/1553374.1553426). In Proceedings of International Conference on Machine Learning (pp. 401-408).

  * Suttorp, T., Hansen, N. and Igel, C., 2009. [Efficient covariance matrix update for variable metric evolution strategies](https://link.springer.com/article/10.1007/s10994-009-5102-1). Machine Learning, 75(2), pp.167-197.

  * Heidrich-Meisner, V. and Igel, C., 2008, September. [Evolution strategies for direct policy search](https://link.springer.com/chapter/10.1007/978-3-540-87700-4_43). In International Conference on Parallel Problem Solving from Nature (pp. 428-437). Springer, Berlin, Heidelberg.

  * Arnold, D.V. and MacLeod, A., 2006, July. [Hierarchically organised evolution strategies on the parabolic ridge](https://dl.acm.org/doi/abs/10.1145/1143997.1144080). In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 437-444). ACM.

  * Igel, C., Suttorp, T. and Hansen, N., 2006, July. [A computational efficient covariance matrix update and a (1+1)-CMA for evolution strategies](https://dl.acm.org/doi/abs/10.1145/1143997.1144082). In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 453-460). ACM.

  * Auger, A. and Hansen, N., 2005, September. [A restart CMA evolution strategy with increasing population size](https://ieeexplore.ieee.org/abstract/document/1554902). In IEEE Congress on Evolutionary Computation (pp. 1769-1776). IEEE.

  * Hansen, N., Müller, S.D. and Koumoutsakos, P., 2003. [Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES)](https://direct.mit.edu/evco/article-abstract/11/1/1/1139/Reducing-the-Time-Complexity-of-the-Derandomized). Evolutionary Computation, 11(1), pp.1-18.

  * Beyer, H.G. and Schwefel, H.P., 2002. [Evolution strategies–A comprehensive introduction](https://link.springer.com/article/10.1023/A:1015059928466). Natural Computing, 1(1), pp.3-52.

  * Hansen, N. and Ostermeier, A., 2001. [Completely derandomized self-adaptation in evolution strategies](https://direct.mit.edu/evco/article-abstract/9/2/159/892/Completely-Derandomized-Self-Adaptation-in). Evolutionary Computation, 9(2), pp.159-195.

  * Hansen, N. and Ostermeier, A., 1996, May. [Adapting arbitrary normal mutation distributions in evolution strategies: The covariance matrix adaptation](https://ieeexplore.ieee.org/abstract/document/542381). In Proceedings of IEEE International Conference on Evolutionary Computation (pp. 312-317). IEEE.

  * Rudolph, G., 1992. [On correlated mutations in evolution strategies](https://ls11-www.cs.tu-dortmund.de/people/rudolph/publications/papers/PPSN92.pdf). In International Conference on Parallel Problem Solving from Nature (pp. 105-114).

  * Rechenberg, I., 1989. [Evolution strategy: Nature’s way of optimization](https://link.springer.com/chapter/10.1007/978-3-642-83814-9_6). In Optimization: Methods and Applications, Possibilities and Limitations (pp. 106-126). Springer, Berlin, Heidelberg.

  * Schwefel, H.P., 1988. [Collective intelligence in evolving systems](https://link.springer.com/chapter/10.1007/978-3-642-73953-8_8). In Ecodynamics (pp. 95-100). Springer, Berlin, Heidelberg.

  * Schwefel, H.P., 1984. [Evolution strategies: A family of non-linear optimization techniques based on imitating some principles of organic evolution](https://link.springer.com/article/10.1007/BF01876146). Annals of Operations Research, 1(2), pp.165-167.

  * Rechenberg, I., 1984. [The evolution strategy. A mathematical model of darwinian evolution](https://link.springer.com/chapter/10.1007/978-3-642-69540-7_13). In Synergetics—from Microscopic to Macroscopic Order (pp. 122-132). Springer, Berlin, Heidelberg.
  
  * Applications: e.g., [Tian&Ha, 2022, EvoStar](https://link.springer.com/chapter/10.1007/978-3-031-03789-4_18); [Anand et al., 2021](https://iopscience.iop.org/article/10.1088/2632-2153/abf3ac); [Nomura et al., 2021, AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17109); [Liu et al., 2019, AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/4345); [Ha&Schmidhuber, 2018, NeurIPS](https://papers.nips.cc/paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html); [Müller&Glasmachers, 2018, PPSN](https://link.springer.com/chapter/10.1007/978-3-319-99259-4_33); [Chrabąszcz et al., 2018, IJCAI](https://www.ijcai.org/proceedings/2018/197); [OpenAI, 2017](https://openai.com/blog/evolution-strategies/); [Zhang et al., 2017, Science](https://www.science.org/doi/10.1126/science.aal5054)

* Eiben, A.E. and Smith, J., 2015. [From evolutionary computation to the evolution of things](https://www.nature.com/articles/nature14544.). Nature, 521(7553), pp.476-482. [ [http://www.evolutionarycomputation.org/](http://www.evolutionarycomputation.org/) ]

  * Miikkulainen, R. and Forrest, S., 2021. [A biological perspective on evolutionary computation](https://www.nature.com/articles/s42256-020-00278-8). Nature Machine Intelligence, 3(1), pp.9-15.

  * Stork, J., Eiben, A.E. and Bartz-Beielstein, T., 2020. [A new taxonomy of global optimization algorithms](https://link.springer.com/article/10.1007/s11047-020-09820-4). Natural Computing, pp.1-24.

  * Beyer, H.G. and Deb, K., 2001. [On self-adaptive features in real-parameter evolutionary algorithms](https://ieeexplore.ieee.org/abstract/document/930314). IEEE Transactions on Evolutionary Computation, 5(3), pp.250-270.

  * Salomon, R., 1998. [Evolutionary algorithms and gradient search: Similarities and differences](https://ieeexplore.ieee.org/abstract/document/728207). IEEE Transactions on Evolutionary Computation, 2(2), pp.45-55.

  * Fogel, D.B., 1998. [Evolutionary computation: The fossil record](https://ieeexplore.ieee.org/book/5263042). IEEE Press.

  * Wolpert, D.H. and Macready, W.G., 1997. [No free lunch theorems for optimization](https://ieeexplore.ieee.org/document/585893). IEEE Transactions on Evolutionary Computation, 1(1), pp.67-82.

  * Bäck, T. and Schwefel, H.P., 1993. [An overview of evolutionary algorithms for parameter optimization](https://direct.mit.edu/evco/article-abstract/1/1/1/1092/An-Overview-of-Evolutionary-Algorithms-for). Evolutionary Computation, 1(1), pp.1-23.

* Schaul, T., Bayer, J., Wierstra, D., Sun, Y., Felder, M., Sehnke, F., Rückstieß, T. and Schmidhuber, J., 2010. [PyBrain](https://jmlr.org/papers/v11/schaul10a.html). Journal of Machine Learning Research, 11(24), pp.743-746.

  * Schaul, T., 2011. [Studies in continuous black-box optimization](https://people.idsia.ch/~schaul/publications/thesis.pdf). Doctoral Dissertation, Technische Universität München.

* Brookes, D., Busia, A., Fannjiang, C., Murphy, K. and Listgarten, J., 2020, July. [A view of estimation of distribution algorithms through the lens of expectation-maximization](https://dl.acm.org/doi/abs/10.1145/3377929.3389938). In Proceedings of Genetic and Evolutionary Computation Conference Companion (pp. 189-190). ACM.

  * Kabán, A., Bootkrajang, J. and Durrant, R.J., 2016. [Toward large-scale continuous EDA: A random matrix theory perspective](https://direct.mit.edu/evco/article-abstract/24/2/255/1016/Toward-Large-Scale-Continuous-EDA-A-Random-Matrix). Evolutionary Computation, 24(2), pp.255-291.

  * Pelikan, M., Hauschild, M.W. and Lobo, F.G., 2015. [Estimation of distribution algorithms](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_45). In Springer Handbook of Computational Intelligence (pp. 899-928). Springer, Berlin, Heidelberg.

  * Dong, W., Chen, T., Tiňo, P. and Yao, X., 2013. [Scaling up estimation of distribution algorithms for continuous optimization](https://ieeexplore.ieee.org/document/6461934). IEEE Transactions on Evolutionary Computation, 17(6), pp.797-822.
  
  * Hauschild, M. and Pelikan, M., 2011. [An introduction and survey of estimation of distribution algorithms](https://www.sciencedirect.com/science/article/abs/pii/S2210650211000435). Swarm and Evolutionary Computation, 1(3), pp.111-128.

  * Larrañaga, P. and Lozano, J.A. eds., 2001. [Estimation of distribution algorithms: A new tool for evolutionary computation](https://link.springer.com/book/10.1007/978-1-4615-1539-5). Springer Science & Business Media.
  
  * Pelikan, M., Goldberg, D.E. and Cantú-Paz, E., 1999, July. [BOA: The Bayesian optimization algorithm](https://dl.acm.org/doi/abs/10.5555/2933923.2933973). In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 525-532). ACM.

  * Mühlenbein, H., 1997. [The equation for response to selection and its use for prediction](https://tinyurl.com/yt78c786).  Evolutionary Computation, 5(3), pp.303-346.

* De Boer, P.T., Kroese, D.P., Mannor, S. and Rubinstein, R.Y., 2005. [A tutorial on the cross-entropy method](https://link.springer.com/article/10.1007/s10479-005-5724-z). Annals of Operations Research, 134(1), pp.19-67.

  * Rubinstein, R.Y. and Kroese, D.P., 2004. [The cross-entropy method: A unified approach to combinatorial optimization, Monte-Carlo simulation, and machine learning](https://link.springer.com/book/10.1007/978-1-4757-4321-0). New York: Springer.

* Bonyadi, M.R. and Michalewicz, Z., 2017. [Particle swarm optimization for single objective continuous space problems: A review](https://direct.mit.edu/evco/article-abstract/25/1/1/1040/Particle-Swarm-Optimization-for-Single-Objective). Evolutionary Computation, 25(1), pp.1-54.

  * Poli, R., Kennedy, J. and Blackwell, T., 2007. [Particle swarm optimization](https://link.springer.com/article/10.1007/s11721-007-0002-0). Swarm Intelligence, 1(1), pp.33-57.

  * Parsopoulos, K.E. and Vrahatis, M.N., 2002. [Recent approaches to global optimization problems through particle swarm optimization](https://link.springer.com/article/10.1023/A:1016568309421). Natural Computing, 1(2), pp.235-306.

  * Eberhart, R.C., Shi, Y. and Kennedy, J., 2001. [Swarm intelligence](https://www.elsevier.com/books/swarm-intelligence/eberhart/978-1-55860-595-4). Elsevier.

* [https://www.jhuapl.edu/SPSA/](https://www.jhuapl.edu/SPSA/)

* Forrest, S., 1993. [Genetic algorithms: Principles of natural selection applied to computation](https://www.science.org/doi/10.1126/science.8346439). Science, 261(5123), pp.872-878.

  * Whitley, D., 1989, December. [The GENITOR algorithm and selection pressure: Why rank-based allocation of reproductive trials is best](https://dl.acm.org/doi/10.5555/93126.93169). In Proceedings of International Conference on Genetic Algorithms (pp. 116-121).

* Wright, M.H., 1996. [Direct search methods: Once scorned, now respectable](https://nyuscholars.nyu.edu/en/publications/direct-search-methods-once-scorned-now-respectable). Pitman Research Notes in Mathematics Series, pp.191-208.

  * Audet, C., Le Digabel, S., Rochon Montplaisir, V. and Tribes, C., 2022. [Algorithm XXXX: NOMAD version 4: Nonlinear optimization with the MADS algorithm](https://dl.acm.org/doi/abs/10.1145/3544489). ACM Transactions on Mathematical Software.

  * Brent, R.P., 2013. [Algorithms for minimization without derivatives](https://maths-people.anu.edu.au/~brent/pub/pub011.html). Courier Corporation.

  * Singer, S. and Nelder, J., 2009. [Nelder-mead algorithm](http://var.scholarpedia.org/article/Nelder-Mead_algorithm). Scholarpedia, 4(7), p.2928.

  * Kolda, T.G., Lewis, R.M. and Torczon, V., 2003. [Optimization by direct search: New perspectives on some classical and modern methods](https://epubs.siam.org/doi/abs/10.1137/S003614450242889). SIAM Review, 45(3), pp.385-482.

  * Lagarias, J.C., Reeds, J.A., Wright, M.H. and Wright, P.E., 1998. [Convergence properties of the Nelder--Mead simplex method in low dimensions](https://epubs.siam.org/doi/abs/10.1137/S1052623496303470). SIAM Journal on Optimization, 9(1), pp.112-147.

  * Powell, M.J., 1998. [Direct search algorithms for optimization calculations](https://www.cambridge.org/core/journals/acta-numerica/article/abs/direct-search-algorithms-for-optimization-calculations/23FA5B19EAF122E02D3724DB1841238C). Acta Numerica, 7, pp.287-336.

  * Torczon, V., 1997. [On the convergence of pattern search algorithms](https://epubs.siam.org/doi/abs/10.1137/S1052623493250780). SIAM Journal on Optimization, 7(1), pp.1-25.

  * Nelder, J.A. and Mead, R., 1965. [A simplex method for function minimization](https://academic.oup.com/comjnl/article-abstract/7/4/308/354237). The Computer Journal, 7(4), pp.308-313.

  * Powell, M.J., 1964. [An efficient method for finding the minimum of a function of several variables without calculating derivatives](https://academic.oup.com/comjnl/article-abstract/7/2/155/335330). The Computer Journal, 7(2), pp.155-162.

  * Kaupe Jr, A.F., 1963. [Algorithm 178: Direct search](https://dl.acm.org/doi/pdf/10.1145/366604.366632). Communications of the ACM, 6(6), pp.313-314.

  * Spang, III, H.A., 1962. [A review of minimization techniques for nonlinear functions](https://epubs.siam.org/doi/abs/10.1137/1004089). SIAM Review, 4(4), pp.343-365.

  * Hooke, R. and Jeeves, T.A., 1961. [“Direct search” solution of numerical and statistical problems](https://dl.acm.org/doi/10.1145/321062.321069). Journal of the ACM, 8(2), pp.212-229. [ [Python - pymoo](https://pymoo.org/algorithms/soo/pattern.html) | [This Week's Citation Classic](http://garfield.library.upenn.edu/classics1980/A1980JK10100001.pdf) ]

  * Box, G.E., 1957. [Evolutionary operation: A method for increasing industrial productivity](https://rss.onlinelibrary.wiley.com/doi/abs/10.2307/2985505). Journal of the Royal Statistical Society: Series C (Applied Statistics), 6(2), pp.81-101.

  * Fermi, E. and Metropolis N., 1952. [Numerical solution of a minimum problem](https://www.osti.gov/servlets/purl/4377177). Los Alamos Scientific Lab., Los Alamos, NM.
  
    * Anderson, H.L., Davidon, W.C., Glicksman, M. and Kruse, U.E., 1955. [Scattering of positive pions by hydrogen at 189 MeV](https://journals.aps.org/pr/abstract/10.1103/PhysRev.100.279). Physical Review, 100(1), p.279.

  * Applications: e.g., [NM: [Sparrow et al., 2018, Nature](https://www.nature.com/articles/s41586-018-0152-9); [Gillon et al., Nature, 2017](https://www.nature.com/articles/nature21360); [Lau&Rubinstein, Nature, 2012](https://www.nature.com/articles/nature10699) ]; [HJ: [Khaledian et al., 2018, IEEE-TMTT](https://ieeexplore.ieee.org/abstract/document/8335770); [Luhar et al., 2015, JFM](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/framework-for-studying-the-effect-of-compliant-surfaces-on-wall-turbulence/47B8C79442109015690B6B16F891DF09); [Paxton et al., 2013, ApJS](https://iopscience.iop.org/article/10.1088/0067-0049/208/1/4); [Schneider, 1999, Genetics](https://academic.oup.com/genetics/article/152/3/1079/6034947); [Ditchfield et al., 1971, JCP](https://aip.scitation.org/doi/10.1063/1.1674902)]

* [https://bayesoptbook.com/](https://bayesoptbook.com/)

  * [https://bayesopt-tutorial.github.io/](https://bayesopt-tutorial.github.io/)

* Press, W.H., Teukolsky, S.A., Vetterling, W.T. and Flannery, B.P., 2007. [Numerical recipes: The art of scientific computing](http://numerical.recipes/). Cambridge University Press. (See Chapter 10. Minimization or maximization of functions.)

  * Custódio, A.L., Scheinberg, K. and Nunes Vicente, L., 2017. [Methodologies and software for derivative-free optimization](https://epubs.siam.org/doi/abs/10.1137/1.9781611974683.ch37). Advances and Trends in Optimization with Engineering Applications, pp.495-506.

## Research Support

This open-source Python library for black-box optimization is now supported by **Shenzhen Fundamental Research Program** under Grant No. JCYJ20200109141235597 (￥2,000,000 from 2021 to 2023), granted to **Prof. Yuhui Shi** (CSE, SUSTech @ Shenzhen, China), and actively developed by three of his group members (e.g., *Qiqi Duan*, *Chang Shao*, *Guochen Zhou*).

Now [Zhuowei Wang](https://scholar.google.com/citations?user=JJb16CAAAAAJ) from University of Technology Sydney (UTS) takes part in this library as one core developer (for testing). Mingyang Feng from University of Birmingham helps to search papers involved in this library.
