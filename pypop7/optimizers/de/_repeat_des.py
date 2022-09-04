"""Repeat the following paper for `DES`:
    Arabas, J. and Jagodziński, D., 2019.
    Toward a matrix-free covariance matrix adaptation evolution strategy.
    IEEE Transactions on Evolutionary Computation, 24(1), pp.84-98.
    https://doi.org/10.1109/TEVC.2019.2907266

    Very close performance can be obtained by our code. Therefore, we argue that the repeatability of
    `DES` can be well-documented (*at least partly*).
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere, cigar, discus, ellipsoid, rosenbrock
from pypop7.optimizers.de.des import DES


if __name__ == '__main__':
    ndim_problem = 30

    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100*np.ones((ndim_problem,)),
               'upper_boundary': 100*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 1000 * ndim_problem,
               # 'n_individuals': 4 * ndim_problem,
               'seed_rng': 0,  # undefined in the original paper
               }
    des = DES(problem, options)
    results = des.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 6617
    # vs 1.0e-10 (from the original paper)

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100*np.ones((ndim_problem,)),
               'upper_boundary': 100*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 2000 * ndim_problem,
               # 'n_individuals': 4 * ndim_problem,
               'seed_rng': 0,  # undefined in the original paper
               }
    des = DES(problem, options)
    results = des.optimize()
    print(results)
    # 224142425
    # vs 1.0e+2 (from the original paper)
