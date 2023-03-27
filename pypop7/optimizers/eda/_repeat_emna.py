"""Repeat the following paper for `EMNA`:
    Larrañaga, P. and Lozano, J.A. eds., 2001.
    Estimation of distribution algorithms: A new tool for evolutionary computation.
    Springer Science & Business Media.
    https://link.springer.com/book/10.1007/978-1-4615-1539-5
    (See Chapter 8 Experimental Results in Function Optimization with EDAs in Continuous Domain.)

    There seems to be vey small performance gaps between the original paper and our Python code, which
    may be ignored for benchmarking. Such gaps may attribute to a slight implementation difference
    (where our Python code does not use *ad-hoc elitist selection* employed in the original paper).
    We argue that its repeatability can be **well-documented**.
"""
import numpy as np

from pypop7.benchmarks.base_functions import rosenbrock, griewank
from pypop7.optimizers.eda.emna import EMNA


if __name__ == '__main__':
    ndim_problem = 10

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -10*np.ones((ndim_problem,)),
               'upper_boundary': 10*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 301850,
               'n_individuals': 2000,
               'seed_rng': 0}  # undefined in the original paper
    emna = EMNA(problem, options)
    results = emna.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 7.674018032114779 vs 8.7201 (from the original paper)

    problem = {'fitness_function': griewank,
               'ndim_problem': ndim_problem,
               'lower_boundary': -600*np.ones((ndim_problem,)),
               'upper_boundary': 600*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 301850,
               'n_individuals': 750,
               'seed_rng': 0}  # undefined in the original paper
    emna = EMNA(problem, options)
    results = emna.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 0.0 vs 5.1166e-02 (from the original paper)

    ndim_problem = 50

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -10*np.ones((ndim_problem,)),
               'upper_boundary': 10*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 301850,
               'n_individuals': 2000,
               'seed_rng': 0}  # undefined in the original paper
    emna = EMNA(problem, options)
    results = emna.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 56.71781067326537 vs 49.7588 (from the original paper)

    problem = {'fitness_function': griewank,
               'ndim_problem': ndim_problem,
               'lower_boundary': -600*np.ones((ndim_problem,)),
               'upper_boundary': 600*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 301850,
               'n_individuals': 750,
               'seed_rng': 0}  # undefined in the original paper
    emna = EMNA(problem, options)
    results = emna.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 1.2459032377053032 vs 8.7673e-06 (from the original paper)
