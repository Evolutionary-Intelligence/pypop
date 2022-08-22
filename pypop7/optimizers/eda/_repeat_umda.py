"""Repeat the following paper:
    Larranaga, P., Etxeberria, R., Lozano, J.A. and Pena, J.M., 2000.
    Optimization in continuous domains by learning and simulation of Gaussian networks.
    Technical Report, Department of Computer Science and Artificial Intelligence,
    University of the Basque Country.
    https://tinyurl.com/3bw6n3x4
"""
import numpy as np

from pypop7.benchmarks.base_functions import rosenbrock, griewank
from pypop7.optimizers.eda.umda import UMDA


if __name__ == '__main__':
    ndim_problem = 10

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -10*np.ones((ndim_problem,)),
               'upper_boundary': 10*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 300000,
               'n_individuals': 2000,
               'seed_rng': 0}  # undefined in the original paper
    umda = UMDA(problem, options)
    results = umda.optimize()
    print(results)
    print(results['best_so_far_y'])  # 8.221260903106916 vs 0.13754 (from the original paper)
    # The performance difference may come from difference of the initialization process

    problem = {'fitness_function': griewank,
               'ndim_problem': ndim_problem,
               'lower_boundary': -600*np.ones((ndim_problem,)),
               'upper_boundary': 600*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 300000,
               'n_individuals': 750,
               'seed_rng': 0}  # undefined in the original paper
    umda = UMDA(problem, options)
    results = umda.optimize()
    print(results)
    print(results['best_so_far_y'])  # 0.0 vs 0.011076 (from the original paper)
