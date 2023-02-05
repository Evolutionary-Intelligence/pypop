"""Repeat the following paper for `VDCMA`:
    Akimoto, Y., Auger, A. and Hansen, N., 2014, July.
    Comparison-based natural gradient optimization in high dimension.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 373-380). ACM.
    https://dl.acm.org/doi/abs/10.1145/2576768.2598258
    https://gist.github.com/youheiakimoto/08b95b52dfbf8832afc71dfff3aed6c8

    Luckily our Python code could repeat the data reported in the official Python code *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import ellipsoid
from pypop7.optimizers.es.vdcma import VDCMA


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 1000
    for f in [ellipsoid]:
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem}
        options = {'fitness_threshold': 1e-8,
                   'seed_rng': 0,
                   'x': 3.0 * np.ones((ndim_problem,)),  # mean
                   'sigma': 1.0,
                   'verbose': 2000,
                   'saving_fitness': 20000}
        vdcma = VDCMA(problem, options)
        print(vdcma.optimize())
        print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
        # 802321 vs 808320 (from the official Python code)
