"""Repeat the following paper for `CODE`:
    Wang, Y., Cai, Z., and Zhang, Q. 2011.
    Differential evolution with composite trial vector generation strategies and control parameters.
    IEEE Transactions on Evolutionary Computation, 15(1), pp.55–66.
    https://doi.org/10.1109/TEVC.2010.2087271

    Very close performance can be obtained by our code. Therefore, we argue that the repeatability of
    `CODE` can be well-documented (*at least partly*).
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere, rosenbrock, rastrigin
from pypop7.optimizers.de.code import CODE


if __name__ == '__main__':
    ndim_problem = 30

    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100*np.ones((ndim_problem,)),
               'upper_boundary': 100*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 10000 * ndim_problem,
               'n_individuals': 30,
               'seed_rng': 0,  # undefined in the original paper
               'fitness_threshold': 1e-8,  # from the original paper
               }
    code = CODE(problem, options)
    results = code.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 0.00
    # vs 0.00 (from the original paper)

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100*np.ones((ndim_problem,)),
               'upper_boundary': 100*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 10000 * ndim_problem,
               'n_individuals': 30,
               'seed_rng': 0,  # undefined in the original paper
               'fitness_threshold': 1e-8,  # from the original paper
               }
    code = CODE(problem, options)
    results = code.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 76.7
    # vs 1.6E-01 (from the original paper)

    problem = {'fitness_function': rastrigin,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5*np.ones((ndim_problem,)),
               'upper_boundary': 5*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 10000 * ndim_problem,
               'n_individuals': 30,
               'seed_rng': 0,  # undefined in the original paper
               'fitness_threshold': 1e-8,  # from the original paper
               }
    code = CODE(problem, options)
    results = code.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 0.00
    # vs 0.00 (from the original paper)
