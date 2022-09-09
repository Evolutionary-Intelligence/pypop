"""Repeat the following paper for `DES`:
    Arabas, J. and Jagodziński, D., 2019.
    Toward a matrix-free covariance matrix adaptation evolution strategy.
    IEEE Transactions on Evolutionary Computation, 24(1), pp.84-98.
    https://doi.org/10.1109/TEVC.2019.2907266
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere, cigar, discus, ellipsoid, rosenbrock
from pypop7.optimizers.de.des import DES


if __name__ == '__main__':
    ndim_problem = 30
    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5*np.ones((ndim_problem,)),
               'upper_boundary': 5*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 8000,
               'seed_rng': 0,  # undefined in the original paper
               'record_fitness': True}
    des = DES(problem, options)
    results = des.optimize()
    print(results)
    print(results['best_so_far_y'])
    #  vs ~1.0e-8 (from the original paper)

    problem = {'fitness_function': cigar,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5*np.ones((ndim_problem,)),
               'upper_boundary': 5*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 14000,
               'seed_rng': 0,  # undefined in the original paper
               'record_fitness': True}
    des = DES(problem, options)
    results = des.optimize()
    print(results)
    print(results['best_so_far_y'])
    #  vs ~1.0e-10 (from the original paper)

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5*np.ones((ndim_problem,)),
               'upper_boundary': 5*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 50000,
               'seed_rng': 0,  # undefined in the original paper
               'record_fitness': True}
    des = DES(problem, options)
    results = des.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 29.87078709080472 vs ~1.0e+2 (from the original paper)

    problem = {'fitness_function': discus,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5 * np.ones((ndim_problem,)),
               'upper_boundary': 5 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 14000,
               'seed_rng': 0,  # undefined in the original paper
               'record_fitness': True}
    des = DES(problem, options)
    results = des.optimize()
    print(results)
    print(results['best_so_far_y'])
    #  vs ~1.0e+2 (from the original paper)

    problem = {'fitness_function': ellipsoid,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5 * np.ones((ndim_problem,)),
               'upper_boundary': 5 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 14000,
               'seed_rng': 0,  # undefined in the original paper
               'record_fitness': True}
    des = DES(problem, options)
    results = des.optimize()
    print(results)
    print(results['best_so_far_y'])
    #  vs ~1.0e+2 (from the original paper)
