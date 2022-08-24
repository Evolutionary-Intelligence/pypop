"""Repeat the following paper for `AEMNA`:
    Larrañaga, P. and Lozano, J.A. eds., 2001.
    Estimation of distribution algorithms: A new tool for evolutionary computation.
    Springer Science & Business Media.
    https://link.springer.com/book/10.1007/978-1-4615-1539-5
    (See Chapter 8 Experimental Results in Function Optimization with EDAs in Continuous Domain.)

    Surprisingly, our code reported *much better* results than the original paper. However, given that it shows
    very similar performance with its same kind `EMNA`, we still argue that the repeatability of `AEMNA` can be
    well-documented (*at least partly*).
"""
import numpy as np

from pypop7.benchmarks.base_functions import rosenbrock, griewank
from pypop7.optimizers.eda.aemna import AEMNA


if __name__ == '__main__':
    ndim_problem = 10

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -10*np.ones((ndim_problem,)),
               'upper_boundary': 10*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 301850,
               'n_individuals': 2000,
               'seed_rng': 0,  # undefined in the original paper
               'verbose_frequency': 1000}
    aemna = AEMNA(problem, options)
    results = aemna.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 7.517279751033313
    # vs 3263.0010 (from the original paper)

    problem = {'fitness_function': griewank,
               'ndim_problem': ndim_problem,
               'lower_boundary': -600*np.ones((ndim_problem,)),
               'upper_boundary': 600*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 301850,
               'n_individuals': 750,
               'seed_rng': 0,  # undefined in the original paper
               'verbose_frequency': 1000}
    aemna = AEMNA(problem, options)
    results = aemna.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 0.7944761831988641
    # vs 12.9407 (from the original paper)
