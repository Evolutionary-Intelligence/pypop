"""Repeat the following paper for `LEP`:
    Lee, C.Y. and Yao, X., 2004.
    Evolutionary programming using mutations based on the Lévy probability distribution.
    IEEE Transactions on Evolutionary Computation, 8(1), pp.1-13.
    https://ieeexplore.ieee.org/document/1266370

    Luckily our Python code could repeat the data reported in the original paper *nearly well*.
    Therefore, we argue that its repeatability could be **well-documented** (*at least partly*).
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere, rosenbrock, rastrigin, ackley
from pypop7.optimizers.ep.lep import LEP


if __name__ == '__main__':
    ndim_problem = 30

    problem = {'fitness_function': ackley,
               'ndim_problem': ndim_problem,
               'lower_boundary': -32*np.ones((ndim_problem,)),
               'upper_boundary': 32*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 1500*100,
               'seed_rng': 0,  # undefined in the original paper
               'sigma': 3.0}
    lep = LEP(problem, options)
    results = lep.optimize()
    print(results['best_so_far_y'])
    # 0.09633170691381965 vs 0.974767

    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100*np.ones((ndim_problem,)),
               'upper_boundary': 100*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 1500*100,
               'seed_rng': 0,  # undefined in the original paper
               'sigma': 3.0}
    lep = LEP(problem, options)
    results = lep.optimize()
    print(results['best_so_far_y'])
    # 0.0062445469828842585 vs 0.001979

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -30*np.ones((ndim_problem,)),
               'upper_boundary': 30*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 1500*100,
               'seed_rng': 0,  # undefined in the original paper
               'sigma': 3.0}
    lep = LEP(problem, options)
    results = lep.optimize()
    print(results['best_so_far_y'])
    # 37.60584627168451 vs 72.343559

    problem = {'fitness_function': rastrigin,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5.12*np.ones((ndim_problem,)),
               'upper_boundary': 5.12*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 1500*100,
               'seed_rng': 0,  # undefined in the original paper
               'sigma': 3.0}
    lep = LEP(problem, options)
    results = lep.optimize()
    print(results['best_so_far_y'])
    # 425.4823087914274 vs 38.266239
