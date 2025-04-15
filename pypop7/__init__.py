"""PyPop7 is a Pure-PYthon library of POPulation-based (e.g., evolutionary/swarm)
    OPtimization for single-objective, real-parameter, global problems. Its main
    goal is to provide a *unified* interface and *elegant* implementations for
    Derivative-Free Optimization (DFO), particularly population-based optimizers,
    in order to facilitate research repeatability, DFO benchmarking, and also
    (possible) real-world applications.

    More specifically, for alleviating the notorious **curse of dimensionality**
    of DFO (based on iterative sampling), one primary focus of PyPop7 is to cover
    their State-Of-The-Art (SOTA) implementations for Large-Scale Optimization
    (LSO), though many (rather all) of their small-and-medium-scaled versions and
    variants are also included for *benchmarking*, *hybrid*, *educational*, and
    more importantly **practical** purposes.
"""
# *base* (*abstract*) class for all global optimizers
from pypop7.optimizers.core import Optimizer, Terminations
