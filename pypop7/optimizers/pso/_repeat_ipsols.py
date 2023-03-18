"""Repeat the following paper for `IPSOLS`:
    De Oca, M.A.M., Stutzle, T., Van den Enden, K. and Dorigo, M., 2010.
    Incremental social learning in particle swarms.
    IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 41(2), pp.368-384.
    https://ieeexplore.ieee.org/document/5582312

    Luckily our code could repeat the data reported in the original paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import step
from pypop7.benchmarks.shifted_functions import sphere, griewank, rastrigin, rosenbrock, salomon, generate_shift_vector
from pypop7.optimizers.pso.ipsols import IPSOLS as Solver


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 100

    generate_shift_vector(sphere, ndim_problem, -100.0, 100.0, 0)
    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100.0*np.ones((ndim_problem,)),
               'upper_boundary': 100.0*np.ones((ndim_problem,))}
    options = {'seed_rng': 0,  # not given in the original paper
               'max_function_evaluations': 1e6}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)  # 0.0e0 vs 0.0e0 (from original paper)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))

    generate_shift_vector(griewank, ndim_problem, -600.0, 600.0, 0)
    problem = {'fitness_function': griewank,
               'ndim_problem': ndim_problem,
               'lower_boundary': -600.0*np.ones((ndim_problem,)),
               'upper_boundary': 600.0*np.ones((ndim_problem,))}
    options = {'seed_rng': 0,  # not given in the original paper
               'max_function_evaluations': 1e6}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)  # 4.66e-9 vs 0.0e0 (from original paper)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))

    generate_shift_vector(rastrigin, ndim_problem, -5.12, 5.12, 0)
    problem = {'fitness_function': rastrigin,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5.12*np.ones((ndim_problem,)),
               'upper_boundary': 5.12*np.ones((ndim_problem,))}
    options = {'seed_rng': 0,  # not given in the original paper
               'max_function_evaluations': 1e6}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)  # 3.33e2 vs 1.53e2 (from original paper)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))

    generate_shift_vector(rosenbrock, ndim_problem, -30.0, 30.0, 0)
    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -30.0*np.ones((ndim_problem,)),
               'upper_boundary': 30.0*np.ones((ndim_problem,))}
    options = {'seed_rng': 0,  # not given in the original paper
               'max_function_evaluations': 1e6}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)  # 0.09637797879698891 vs 8.89e-3 (from original paper)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))

    problem = {'fitness_function': step,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5.12*np.ones((ndim_problem,)),
               'upper_boundary': 5.12*np.ones((ndim_problem,))}
    options = {'seed_rng': 0,  # not given in the original paper
               'max_function_evaluations': 1e6}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)  # 0.0e0 vs 0.0e0 (from original paper)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))

    generate_shift_vector(salomon, ndim_problem, -100.0, 100.0, 0)
    problem = {'fitness_function': salomon,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100.0*np.ones((ndim_problem,)),
               'upper_boundary': 100.0*np.ones((ndim_problem,))}
    options = {'seed_rng': 0,  # not given in the original paper
               'verbose': 1,
               'max_function_evaluations': 1e6}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)  # 1.49e1 vs 4.7e0 (from original paper)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
