"""Repeat the following paper for `FEP`:
    Yao, X., Liu, Y. and Lin, G., 1999.
    Evolutionary programming made faster.
    IEEE Transactions on Evolutionary Computation, 3(2), pp.82-102.
    https://ieeexplore.ieee.org/abstract/document/771163

    Since its source code is not openly available, the performance differences between our code and the original
    paper are very hard (if not impossible) to analyze. We notice that different people may give a different
    implementation based on different understandings of its algorithmic operations.

    Unfortunately, till now our code *cannot* perfectly repeat the results reported by the original paper.
    We expect that a much closer open-source implementation could be given in the future.
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere, step, rosenbrock, rastrigin, ackley
from pypop7.optimizers.ep.fep import FEP


if __name__ == '__main__':
    ndim_problem = 30

    problem = {'fitness_function': rastrigin,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5.12 * np.ones((ndim_problem,)),
               'upper_boundary': 5.12 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 5000 * 100,
               'seed_rng': 0,  # undefined in the original paper
               'sigma': 3.0}
    fep = FEP(problem, options)
    results = fep.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 15.919371683065265
    # vs 4.6e-2 (from the original paper)

    problem = {'fitness_function': ackley,
               'ndim_problem': ndim_problem,
               'lower_boundary': -32 * np.ones((ndim_problem,)),
               'upper_boundary': 32 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 1500 * 100,
               'seed_rng': 0,  # undefined in the original paper
               'sigma': 3.0}
    fep = FEP(problem, options)
    results = fep.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 1.8248431332886352
    # vs 1.8e-2 (from the original paper)

    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100*np.ones((ndim_problem,)),
               'upper_boundary': 100*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 1500 * 100,
               'seed_rng': 0,  # undefined in the original paper
               'sigma': 3.0}
    fep = FEP(problem, options)
    results = fep.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 0.014982130779192806
    # vs 5.7e-4 (from the original paper)

    problem = {'fitness_function': step,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100 * np.ones((ndim_problem,)),
               'upper_boundary': 100 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 1500 * 100,
               'seed_rng': 0,  # undefined in the original paper
               'sigma': 3.0}
    fep = FEP(problem, options)
    results = fep.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 34.0
    # vs 0 (from the original paper)

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -30*np.ones((ndim_problem,)),
               'upper_boundary': 30*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 20000 * 100,
               'seed_rng': 1,  # undefined in the original paper
               'sigma': 3.0}
    fep = FEP(problem, options)
    results = fep.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 21.19095109757174
    # vs mean 5.06 std 5.87 (from the original paper)
