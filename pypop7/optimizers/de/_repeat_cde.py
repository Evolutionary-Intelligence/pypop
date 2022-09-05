"""Repeat the following paper for `CDE`:
    Storn, R.M. and Price, K.V. 1997.
    Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces.
    Journal of Global Optimization, 11(4), pp.341–359.
    https://link.springer.com/article/10.1023/A:1008202821328

    Very close performance can be obtained by our code. Therefore, we argue that
    the repeatability of `CDE` can be well-documented (*at least partly*).
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere, rosenbrock, griewank
from pypop7.optimizers.de.cde import CDE


if __name__ == '__main__':
    problem = {'fitness_function': sphere,
               'ndim_problem': 3,
               'lower_boundary': -5.12 * np.ones((3,)),
               'upper_boundary': 5.12 * np.ones((3,))}
    options = {'fitness_threshold': 1e-6,
               'seed_rng': 0,
               'n_individuals': 5,
               'f': 0.9,
               'cr': 0.1}
    cde = CDE(problem, options)
    results = cde.optimize()
    print(results['n_function_evaluations'], results['best_so_far_y'])
    # 343 vs 406 (from the original paper)

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': 2,
               'lower_boundary': -2.048 * np.ones((2,)),
               'upper_boundary': 2.048 * np.ones((2,))}
    options = {'fitness_threshold': 1e-6,
               'seed_rng': 0,
               'n_individuals': 10,
               'f': 0.9,
               'cr': 0.9}
    cde = CDE(problem, options)
    results = cde.optimize()
    print(results['n_function_evaluations'], results['best_so_far_y'])
    # 579 vs 654 (from the original paper)

    problem = {'fitness_function': griewank,
               'ndim_problem': 10,
               'lower_boundary': -400 * np.ones((10,)),
               'upper_boundary': 400 * np.ones((10,))}
    options = {'fitness_threshold': 1e-6,
               'seed_rng': 0,
               'n_individuals': 25,
               'f': 0.5,
               'cr': 0.2}
    cde = CDE(problem, options)
    results = cde.optimize()
    print(results['n_function_evaluations'], results['best_so_far_y'])
    # 12265 vs 12752 (from the original paper)
