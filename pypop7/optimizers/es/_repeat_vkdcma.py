"""Repeat the following paper for `VKDCMA`:
    https://gist.github.com/youheiakimoto/2fb26c0ace43c22b8f19c7796e69e108

    Akimoto, Y. and Hansen, N., 2016, September.
    Online model selection for restricted covariance matrix adaptation.
    In Parallel Problem Solving from Nature. Springer International Publishing.
    https://link.springer.com/chapter/10.1007/978-3-319-45823-6_1

    Akimoto, Y. and Hansen, N., 2016, July.
    Projection-based restricted covariance matrix adaptation for high dimension.
    In Proceedings of Annual Genetic and Evolutionary Computation Conference 2016 (pp. 197-204). ACM.
    https://dl.acm.org/doi/abs/10.1145/2908812.2908863

    Luckily our Python code could repeat the data reported in the official Python code *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import ellipsoid
from pypop7.optimizers.es.vkdcma import VKDCMA


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 2000
    for f in [ellipsoid]:
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem}
        options = {'fitness_threshold': 1e-8,
                   'seed_rng': 1,
                   'x': 3.0 * np.ones((ndim_problem,)),  # mean
                   'sigma': 2.0,
                   'verbose': 2000,
                   'saving_fitness': 20000}
        vkdcma = VKDCMA(problem, options)
        print(vkdcma.optimize())
        print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
        # 1.271882e+06 vs 1270880 (from the official Python code)
