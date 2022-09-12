"""Repeat the following paper for `ASGA`:
    Demo, N., Tezzele, M. and Rozza, G., 2021.
    A supervised learning approach involving active subspaces for an efficient genetic algorithm in high-dimensional
    optimization problems.
    SIAM Journal on Scientific Computing, 43(3), pp.B831-B853.
    https://epubs.siam.org/doi/10.1137/20M1345219

    Very close performance can be obtained by our code. Therefore, we argue that
    the repeatability of `JADE` can be well-documented (*at least partly*).
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere, step, rosenbrock, rastrigin, ackley
from pypop7.optimizers.ga.asga import ASGA


if __name__ == '__main__':
    ndim_problem = 15

    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100*np.ones((ndim_problem,)),
               'upper_boundary': 100*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 1500 * 20,
               'n_initial_individuals': 2000,
               'n_individuals': 200,
               'seed_rng': 0,  # undefined in the original paper
               }
    asga = ASGA(problem, options)
    results = asga.optimize()
    print(results)
    print(results['best_so_far_y'])
    # evaluations 3800
    # 3.07029e-20
    # vs (from the original paper)
