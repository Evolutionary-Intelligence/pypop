"""Repeat the following paper for `CODE`:
    Wang, Y., Cai, Z., and Zhang, Q. 2011.
    Differential evolution with composite trial vector generation strategies and control parameters.
    IEEE Transactions on Evolutionary Computation, 15(1), pp.55–66.
    https://ieeexplore.ieee.org/document/5688232/

    Luckily our Python code could repeat the data reported in the original paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere, rastrigin
from pypop7.optimizers.de.code import CODE


if __name__ == '__main__':
    ndim_problem = 30

    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100*np.ones((ndim_problem,)),
               'upper_boundary': 100*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 10000*ndim_problem,  # 300000
               'n_individuals': 30,
               'seed_rng': 0}
    code = CODE(problem, options)
    results = code.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 2.497959734279275e-23 vs 0.00 (from the original paper)

    problem = {'fitness_function': rastrigin,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5*np.ones((ndim_problem,)),
               'upper_boundary': 5*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 10000*ndim_problem,  # 300000
               'n_individuals': 30,
               'seed_rng': 0}
    code = CODE(problem, options)
    results = code.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 1.1368683772161603e-13 vs 0.00 (from the original paper)
