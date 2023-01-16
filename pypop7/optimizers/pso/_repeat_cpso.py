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

from pypop7.benchmarks.base_functions import rosenbrock, sphere, ackley, rastrigin, griewank
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
               'n_individuals': 10,
               'fitness_threshold': 0.000100}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)  # 39.413944097899126 vs 7.58e-03 (from original paper)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))

    # problem = {'fitness_function': sphere,
    #            'ndim_problem': ndim_problem,
    #            'lower_boundary': -100 * np.ones((ndim_problem,)),
    #            'upper_boundary': 100 * np.ones((ndim_problem,))}
    # options = {'seed_rng': 0,  # not given in the original paper
    #            'max_function_evaluations': 2e5,
    #            'n_individuals': 10,
    #            'fitness_threshold': 0.01}
    # solver = Solver(problem, options)
    # results = solver.optimize()
    # print(results)  # 3.6076672545234767 vs  (from original paper)
    # print('*** Runtime: {:7.5e}'.format(time.time() - start_run))

    # problem = {'fitness_function': ackley,
    #            'ndim_problem': ndim_problem,
    #            'lower_boundary': -30 * np.ones((ndim_problem,)),
    #            'upper_boundary': 30 * np.ones((ndim_problem,))}
    # options = {'seed_rng': 0,  # not given in the original paper
    #            'max_function_evaluations': 2e5,
    #            'n_individuals': 10,
    #            'fitness_threshold': 5.00}
    # solver = Solver(problem, options)
    # results = solver.optimize()
    # print(results)  # vs (from original paper)
    # print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
    #
    # problem = {'fitness_function': rastrigin,
    #            'ndim_problem': ndim_problem,
    #            'lower_boundary': -5.12 * np.ones((ndim_problem,)),
    #            'upper_boundary': 5.12 * np.ones((ndim_problem,))}
    # options = {'seed_rng': 0,  # not given in the original paper
    #            'max_function_evaluations': 2e5,
    #            'n_individuals': 10,
    #            'fitness_threshold': 100}
    # solver = Solver(problem, options)
    # results = solver.optimize()
    # print(results)  # vs (from original paper)
    # print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
    #
    # problem = {'fitness_function': griewank,
    #            'ndim_problem': ndim_problem,
    #            'lower_boundary': -600 * np.ones((ndim_problem,)),
    #            'upper_boundary': 600 * np.ones((ndim_problem,))}
    # options = {'seed_rng': 0,  # not given in the original paper
    #            'max_function_evaluations': 2e5,
    #            'n_individuals': 10,
    #            'fitness_threshold': 0.1}
    # solver = Solver(problem, options)
    # results = solver.optimize()
    # print(results)  # vs (from original paper)
    # print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
