"""Repeat Table 2 and 3 from the following paper:
    Ros, R. and Hansen, N., 2008, September.
    A simple modification in CMA-ES achieving linear time and space complexity.
    In International Conference on Parallel Problem Solving from Nature (pp. 296-305).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-540-87700-4_30
"""
import numpy as np

from benchmarks.base_functions import rosenbrock
from optimizers.es.sepcmaes import SEPCMAES


if __name__ == '__main__':
    # Repeat Table 2
    ndim_problem = 30
    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem}
    options = {'max_function_evaluations': 200 * 1e3,
               'fitness_threshold': 1e-6,
               'seed_rng': 2022,  # not given in the original paper
               'x': np.zeros((ndim_problem,)),
               'sigma': 0.1,
               'verbose_frequency': 1000,
               'record_fitness': True,
               'record_fitness_frequency': 1000}
    solver = SEPCMAES(problem, options)
    results = solver.optimize()
    print(results)
    print(results['n_function_evaluations'])  # 102590
    # Repeat Table 3
    ndim_problem = 20
    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem}
    options = {'max_function_evaluations': 200 * 1e3,
               'fitness_threshold': 1e-9,
               'seed_rng': 2022,  # not given in the original paper
               'x': np.zeros((ndim_problem,)),
               'sigma': 0.1,
               'verbose_frequency': 1000,
               'record_fitness': True,
               'record_fitness_frequency': 1000,
               'stagnation': np.Inf}
    solver = SEPCMAES(problem, options)
    results = solver.optimize()
    print(results)
    print(results['n_function_evaluations'])  # 110151
