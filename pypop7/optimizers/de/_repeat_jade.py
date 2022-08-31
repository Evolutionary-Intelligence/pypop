"""Repeat the following paper for `JADE`:
    Zhang, J., and Sanderson, A. C. 2009.
    JADE: Adaptive differential evolution with optional external archive.
    IEEE Transactions on Evolutionary Computation, 13(5), 945–958.
    https://doi.org/10.1109/TEVC.2009.2014613

    Very close performance can be obtained by our code. Therefore, we argue that the repeatability of
    `JADE` can be well-documented (*at least partly*).
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere, step, rosenbrock, rastrigin, ackley
from pypop7.optimizers.de.jade import JADE


if __name__ == '__main__':
    ndim_problem = 30

    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100*np.ones((ndim_problem,)),
               'upper_boundary': 100*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 5000 * 100,
               'n_individuals': 100,
               'seed_rng': 0,  # undefined in the original paper
               }
    jade = JADE(problem, options)
    results = jade.optimize()
    print(results)
    print(results['best_so_far_y'])
    # generation 1500
    #            6.7E-62
    # vs         1.3E-54 (from the original paper)

    problem = {'fitness_function': step,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100*np.ones((ndim_problem,)),
               'upper_boundary': 100*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 5000 * 100,
               'n_individuals': 100,
               'seed_rng': 0,  # undefined in the original paper
               }
    jade = JADE(problem, options)
    results = jade.optimize()
    print(results)
    print(results['best_so_far_y'])
    # generation 100 1500
    #            6.0 0
    # vs         5.6 0 (from the original paper)

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -30*np.ones((ndim_problem,)),
               'upper_boundary': 30*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 5000 * 100,
               'n_individuals': 100,
               'seed_rng': 0,  # undefined in the original paper
               }
    jade = JADE(problem, options)
    results = jade.optimize()
    print(results)
    print(results['best_so_far_y'])
    # generation 3000    20000
    #            7.6E-05
    # vs         3.2E-01 3.2E-01 (from the original paper)

    problem = {'fitness_function': rastrigin,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5.21*np.ones((ndim_problem,)),
               'upper_boundary': 5.21*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 5000 * 100,
               'n_individuals': 100,
               'seed_rng': 0,  # undefined in the original paper
               }
    jade = JADE(problem, options)
    results = jade.optimize()
    print(results)
    print(results['best_so_far_y'])
    # generation 1000    5000
    #            6.1E-02 0
    # vs         1.4E-04 0 (from the original paper)

    problem = {'fitness_function': ackley,
               'ndim_problem': ndim_problem,
               'lower_boundary': -32*np.ones((ndim_problem,)),
               'upper_boundary': 32*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 5000 * 100,
               'n_individuals': 100,
               'seed_rng': 0,  # undefined in the original paper
               }
    jade = JADE(problem, options)
    results = jade.optimize()
    print(results)
    print(results['best_so_far_y'])
    # generation 500     2000
    #            1.9E-09 7.5E-15
    # vs         3.0E-09 4.4E-15 (from the original paper)

