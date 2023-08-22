"""Repeat the following paper for `CDE`:
    Storn, R.M. and Price, K.V. 1997.
    Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces.
    Journal of Global Optimization, 11(4), pp.341–359.
    https://link.springer.com/article/10.1023/A:1008202821328

    Luckily our Python code could repeat the data reported in the original paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import numpy as np
import time

from pypop7.benchmarks.base_functions import sphere, rosenbrock, griewank
from cde import CDE


if __name__ == '__main__':
    problem = {'fitness_function': sphere,
               'ndim_problem': 3,
               'lower_boundary': -5.12*np.ones((3,)),
               'upper_boundary': 5.12*np.ones((3,))}
    options = {'fitness_threshold': 1e-6,
               'seed_rng': 2,
               'n_individuals': 5,
               'f': 0.9,
               'cr': 0.1}
    start_time = time.time()
    cde = CDE(problem, options)
    results = cde.optimize()
    print(results['n_function_evaluations'], results['best_so_far_y'], time.time() - start_time)
    # 364 vs 406 (from the original paper) 0.03568911552429199

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': 2,
               'lower_boundary': -2.048*np.ones((2,)),
               'upper_boundary': 2.048*np.ones((2,))}
    options = {'fitness_threshold': 1e-6,
               'seed_rng': 1,
               'n_individuals': 10,
               'f': 0.9,
               'cr': 0.9}
    start_time = time.time()
    cde = CDE(problem, options)
    results = cde.optimize()
    print(results['n_function_evaluations'], results['best_so_far_y'], time.time() - start_time)
    # 528 vs 654 (from the original paper) 0.05886220932006836

    problem = {'fitness_function': griewank,
               'ndim_problem': 10,
               'lower_boundary': -400*np.ones((10,)),
               'upper_boundary': 400*np.ones((10,))}
    options = {'fitness_threshold': 1e-6,
               'seed_rng': 0,
               'n_individuals': 25,
               'f': 0.5,
               'cr': 0.2}
    start_time = time.time()
    cde = CDE(problem, options)
    results = cde.optimize()
    print(results['n_function_evaluations'], results['best_so_far_y'], time.time() - start_time)
    # 11554 vs 12752 (from the original paper) 1.1777536869049072
