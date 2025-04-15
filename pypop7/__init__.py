"""PyPop7 is a Pure-PYthon library of POPulation-based (e.g., evolutionary/swarm) OPtimization
    for single-objective, real-parameter, black-box problems. Its main goal is to provide a
    *unified* interface and *elegant* implementations for Black-Box Optimization (BBO),
    particularly population-based optimizers, in order to facilitate research repeatability,
    BBO benchmarking, and also (possible) real-world applications.

    More specifically, for alleviating the notorious **curse of dimensionality** of BBO (based
    on iterative sampling), one primary focus of PyPop7 is to cover their State-Of-The-Art
    (SOTA) implementations for Large-Scale Optimization (LSO), though many of their other
    versions and variants are also included here (for benchmarking/mixing purpose, and also
    for practical purpose).
"""
# *base* (*abstract*) class for all global optimizers
from pypop7.optimizers.core import Optimizer, Terminations
