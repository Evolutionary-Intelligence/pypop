"""Repeat the following paper for `GL25`:
    García-Martínez, C., Lozano, M., Herrera, F., Molina, D. and Sánchez, A.M., 2008.
    Global and local real-coded genetic algorithms based on parent-centric crossover operators.
    European Journal of Operational Research, 185(3), pp.1088-1113.
    https://www.sciencedirect.com/science/article/abs/pii/S0377221706006308

    Luckily our code could repeat the data reported in the original paper *well*.
    Therefore, we argue that the repeatability of `GL25` could be **well-documented**.
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere, schwefel222
from pypop7.optimizers.ga.gl25 import GL25 as Solver


if __name__ == '__main__':
    ndim_problem = 25

    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem,
               'initial_lower_boundary': 4 * np.ones((ndim_problem,)),
               'initial_upper_boundary': 5 * np.ones((ndim_problem,)),
               'lower_boundary': -5.12 * np.ones((ndim_problem,)),
               'upper_boundary': 5.12 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 100000,
               'seed_rng': 0,
               'saving_fitness': 1}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 3.683380983946287e-151 vs 4.36e-147 (from the original paper)

    problem = {'fitness_function': schwefel222,
               'ndim_problem': ndim_problem,
               'initial_lower_boundary': 8 * np.ones((ndim_problem,)),
               'initial_upper_boundary': 10 * np.ones((ndim_problem,)),
               'lower_boundary': -10 * np.ones((ndim_problem,)),
               'upper_boundary': 10 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 100000,
               'seed_rng': 0,
               'saving_fitness': 1}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 2.8901400348477053e-79 vs 9.85e-77 (from the original paper)
