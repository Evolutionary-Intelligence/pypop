"""Repeat the following paper for `ASGA`:
    Demo, N., Tezzele, M. and Rozza, G., 2021.
    A supervised learning approach involving active subspaces for an efficient genetic algorithm in
      high-dimensional optimization problems.
    SIAM Journal on Scientific Computing, 43(3), pp.B831-B853.
    https://epubs.siam.org/doi/10.1137/20M1345219

    Luckily our code could repeat the data reported in the original paper *well*.
    Therefore, we argue that the repeatability of `ASGA` could be **well-documented**.
"""
import numpy as np

from pypop7.benchmarks.base_functions import rosenbrock, ackley
from pypop7.optimizers.ga.asga import ASGA


if __name__ == '__main__':
    ndim_problem = 15
    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5 * np.ones((ndim_problem,)),
               'upper_boundary': 100 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 2000 + 200 * 30,
               'n_initial_individuals': 2000,
               'n_individuals': 200,
               'seed_rng': 0,
               'verbose': 1}
    asga = ASGA(problem, options)
    results = asga.optimize()
    print(results)
    print(results['best_so_far_y'])  # 14.03062287458637

    problem = {'fitness_function': ackley,
               'ndim_problem': ndim_problem,
               'lower_boundary': -15 * np.ones((ndim_problem,)),
               'upper_boundary': 30 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 2000 + 200 * 30,
               'n_initial_individuals': 2000,
               'n_individuals': 200,
               'seed_rng': 0,
               'verbose': 1}
    asga = ASGA(problem, options)
    results = asga.optimize()
    print(results)
    print(results['best_so_far_y'])  # 0.012111557779814763
