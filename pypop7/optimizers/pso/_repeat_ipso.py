"""Repeat the following paper for `IPSO`:
    De Oca, M.A.M., Stutzle, T., Van den Enden, K. and Dorigo, M., 2010.
    Incremental social learning in particle swarms.
    IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 41(2), pp.368-384.
    https://ieeexplore.ieee.org/document/5582312

    Luckily our code could repeat the data reported in the original paper *well*.
    Therefore, we argue that the repeatability of `IPSO` could be **well-documented**.
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import sphere, griewank, rastrigin
from pypop7.optimizers.pso.ipso import IPSO as Solver


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 100

    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100 * np.ones((ndim_problem,)),
               'upper_boundary': 100 * np.ones((ndim_problem,))}
    options = {'seed_rng': 0,  # not given in the original paper
               'max_function_evaluations': 1e6}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)  # 6.4417698062795075e-12 vs 2.82e-11 (from original paper)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))

    problem = {'fitness_function': griewank,
               'ndim_problem': ndim_problem,
               'lower_boundary': -600 * np.ones((ndim_problem,)),
               'upper_boundary': 600 * np.ones((ndim_problem,))}
    options = {'seed_rng': 0,  # not given in the original paper
               'max_function_evaluations': 1e6}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)  # 0.007396040374509694 vs 7.4e-3 (from original paper)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))

    problem = {'fitness_function': rastrigin,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5.12 * np.ones((ndim_problem,)),
               'upper_boundary': 5.12 * np.ones((ndim_problem,))}
    options = {'seed_rng': 0,  # not given in the original paper
               'max_function_evaluations': 1e6}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)  # 163.2439902619535 vs 2.99e2 (from original paper)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
