"""Repeat the following paper for `CPSO`:
    Van den Bergh, F. and Engelbrecht, A.P., 2004.
    A cooperative approach to particle swarm optimization.
    IEEE Transactions on Evolutionary Computation, 8(3), pp.225-239.
    https://ieeexplore.ieee.org/document/1304845

    Luckily our code could repeat the data reported in the original paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import rosenbrock, griewank
from pypop7.optimizers.pso.cpso import CPSO as Solver


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 30

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -2.048 * np.ones((ndim_problem,)),
               'upper_boundary': 2.048 * np.ones((ndim_problem,))}
    options = {'seed_rng': 0,  # not given in the original paper
               'max_function_evaluations': 2e5,
               'fitness_threshold': 0.0}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)  # 2.11884e-1 vs 9.06e-1 (from original paper)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))

    problem = {'fitness_function': griewank,
               'ndim_problem': ndim_problem,
               'lower_boundary': -600 * np.ones((ndim_problem,)),
               'upper_boundary':600 * np.ones((ndim_problem,))}
    options = {'seed_rng': 0,  # not given in the original paper
               'max_function_evaluations': 2e5,
               'fitness_threshold': 0.0}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)  # 1.23742e-2 vs 2.25e-2 (from original paper)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
