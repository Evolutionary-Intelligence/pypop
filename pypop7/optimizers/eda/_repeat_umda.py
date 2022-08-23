"""Repeat the following paper for `UMDA`:
    Larranaga, P., Etxeberria, R., Lozano, J.A. and Pena, J.M., 2000.
    Optimization in continuous domains by learning and simulation of Gaussian networks.
    Technical Report, Department of Computer Science and Artificial Intelligence,
    University of the Basque Country.
    https://tinyurl.com/3bw6n3x4

    Since its source code is not openly available, the performance differences are very hard (
    if not impossible) to analyze.

    However, we again notice the following paper:
    Larrañaga, P. and Lozano, J.A. eds., 2001.
    Estimation of distribution algorithms: A new tool for evolutionary computation.
    Springer Science & Business Media.
    https://link.springer.com/book/10.1007/978-1-4615-1539-5
    (See Chapter 8 Experimental Results in Function Optimization with EDAs in Continuous Domain)

    With nearly the same settings with the first paper, very close performance can be obtained by our code.
    Therefore, we argue that the repeatability of `UMDA` can be well-documented (*at least partly*).
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
    print(results['best_so_far_y'])
    # 8.221260903106916
    # vs 0.13754 (from the first original paper)
    # vs 8.7204  (from the second original paper)

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
    print(results['best_so_far_y'])
    # 0.0
    # vs 0.011076   (from the first original paper)
    # vs 6.0783e-02 (from the second original paper)

    ndim_problem = 50

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -10 * np.ones((ndim_problem,)),
               'upper_boundary': 10 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 300000,
               'n_individuals': 2000,
               'seed_rng': 0}  # undefined in the original paper
    umda = UMDA(problem, options)
    results = umda.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 47.805237128504224
    # vs 48.8949  (from the second original paper)

    problem = {'fitness_function': griewank,
               'ndim_problem': ndim_problem,
               'lower_boundary': -600 * np.ones((ndim_problem,)),
               'upper_boundary': 600 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 300000,
               'n_individuals': 750,
               'seed_rng': 0}  # undefined in the original paper
    umda = UMDA(problem, options)
    results = umda.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 0.0
    # vs 8.9869e-06 (from the second original paper)
