"""PyPop7 is a Pure-PYthon library of POPulation-based OPtimization for single-objective, real-parameter, black-box
    problems (currently still actively developed). Its main goal is to provide a unified interface and elegant
    implementations for Black-Box Optimization (BBO), particularly population-based optimizers, in order to
    facilitate research repeatability and also real-world applications.

    More specifically, for alleviating the notorious curse of dimensionality of BBO (based on iterative sampling),
    the primary focus of PyPop7 is to cover their State-Of-The-Art (SOTA) implementations for Large-Scale Optimization
    (LSO), though many of their other versions and variants are also included here (for benchmarking/mixing purpose,
    and sometimes even for practical purpose).
"""
from pypop7.optimizers.core import Optimizer, Terminations  # base (abstract) class for all optimizers
from pypop7.optimizers.rs import RS, PRS, RHC, ARHC, SRS, BES
from pypop7.optimizers.ep import EP, CEP, FEP, LEP
from pypop7.optimizers.ga import GA, GENITOR, G3PCX, GL25  # ASGA
from pypop7.optimizers.sa import SA, CSA, ESA, NSA
from pypop7.optimizers.cc import CC, COEA, COSYNE
from pypop7.optimizers.pso import PSO, SPSO, SPSOL, CLPSO, IPSO, CCPSO2
from pypop7.optimizers.de import DE, CDE, TDE, JADE, CODE
from pypop7.optimizers.cem import CEM, SCEM, DSCEM, MRAS  # DCEM
from pypop7.optimizers.es import ES, RES
