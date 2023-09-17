"""Repeat the following paper for `SHADE`:
    Tanabe, R. and Fukunaga, A., 2013, June.
    Success-history based parameter adaptation for differential evolution.
    In IEEE Congress on Evolutionary Computation (pp. 71-78). IEEE.
    https://ieeexplore.ieee.org/document/6557555

    Luckily our Python code could repeat the data reported in the original paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere, ellipsoid, cigar, discus
from pypop7.optimizers.de.shade import SHADE


if __name__ == '__main__':
    ndim_problem = 30

    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100*np.ones((ndim_problem,)),
               'upper_boundary': 100*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': ndim_problem*10000,
               'n_individuals': 100,
               'seed_rng': 0,  # undefined in the original paper
               'fitness_threshold': 1e-8}
    shade = SHADE(problem, options)
    results = shade.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 8.957042714361382e-09 vs 0.00e+00 (from the original paper)

    problem = {'fitness_function': ellipsoid,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100*np.ones((ndim_problem,)),
               'upper_boundary': 100*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': ndim_problem*10000,
               'n_individuals': 100,
               'seed_rng': 0,  # undefined in the original paper
               'fitness_threshold': 1e-8}
    shade = SHADE(problem, options)
    results = shade.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 9.642985936066791e-09 vs 9.00e+03 (from the original paper)

    problem = {'fitness_function': cigar,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100*np.ones((ndim_problem,)),
               'upper_boundary': 100*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': ndim_problem*10000,
               'n_individuals': 100,
               'seed_rng': 0,  # undefined in the original paper
               'fitness_threshold': 1e-8}
    shade = SHADE(problem, options)
    results = shade.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 8.657677766913165e-09 vs 4.02e+01 (from the original paper)

    problem = {'fitness_function': discus,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100*np.ones((ndim_problem,)),
               'upper_boundary': 100*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': ndim_problem*10000,
               'n_individuals': 100,
               'seed_rng': 0,  # undefined in the original paper
               'fitness_threshold': 1e-8}
    shade = SHADE(problem, options)
    results = shade.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 9.982659193421452e-09 vs 1.92e-04 (from the original paper)
