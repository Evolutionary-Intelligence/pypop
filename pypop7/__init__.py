"""PyPop7 is a Pure-PYthon library of POPulation-based randomized OPtimization
    algorithm for single-objective, real-parameter, unconstrained problems.
    Its main goal is to provide a *unified* interface as well as *elegant*
    implementations for derivative-free and black-box optimization,
    particularly population-based optimizers, in order to facilitate research
    repeatability, algorithmic benchmarking, and also real-world applications.

    More specifically, for alleviating the notorious **curse of dimensionality**
    issue (owing to a limited population of randomized samples), one *primary*
    focus of PyPop7 is to cover their State-Of-The-Art (SOTA) implementations
    for Large-Scale Optimization (LSO), though many (rather all) of their
    small-and-medium-scaled versions and variants are also included for
    *benchmarking*, *hybrid*, *educational*, and more importantly **practical**
    purposes.
"""
# termination conditions shared by all optimizers
from pypop7.optimizers.core import Terminations
# *base* (*abstract*) class for all optimizers
from pypop7.optimizers.core import Optimizer
