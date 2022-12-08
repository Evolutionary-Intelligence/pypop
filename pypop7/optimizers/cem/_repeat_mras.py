"""Repeat the following paper for `MRAS`:
    Hu, J., Fu, M.C. and Marcus, S.I., 2007.
    A model reference adaptive search method for global optimization.
    Operations Research, 55(3), pp.549-568.
    https://pubsonline.informs.org/doi/abs/10.1287/opre.1060.0367

    Since its source code is not openly available till now, the performance differences between our code and
    the original paper are very hard (if not impossible) to analyze. We notice that it can generate *zero*
    weights for Monte-Carlo estimation of mean and std (in the second generation) to bias the search toward
    the *origin*, leading to very fast fitness decrease in the beginning stage, according to the given setting
    (i.e., `r=10^-4`).

    We still expect that a much closer open-source implementation could be given in the future, no matter by
    ourselves or others.
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import rosenbrock, griewank
from pypop7.optimizers.cem.mras import MRAS as Solver


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 20

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -50 * np.ones((ndim_problem,)),
               'upper_boundary': 50 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 400000,
               'seed_rng': 1,
               'sigma': np.sqrt(500),
               'verbose': 20,
               'saving_fitness': 2000}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)  # 1.88886675e+01
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))

    problem = {'fitness_function': griewank,
               'ndim_problem': ndim_problem,
               'lower_boundary': -50 * np.ones((ndim_problem,)),
               'upper_boundary': 50 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 400000,
               'seed_rng': 0,
               'sigma': np.sqrt(500),
               'verbose': 20,
               'saving_fitness': 2000}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)  # 2.49050070e+00
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
