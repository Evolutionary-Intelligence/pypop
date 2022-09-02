"""Repeat the following paper for `JADE`:
    Zhang, J., and Sanderson, A. C. 2009.
    JADE: Adaptive differential evolution with optional external archive.
    IEEE Transactions on Evolutionary Computation, 13(5), 945–958.
    https://doi.org/10.1109/TEVC.2009.2014613

    Very close performance can be obtained by our code. Therefore, we argue that
    the repeatability of `JADE` can be well-documented (*at least partly*).
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
    options = {'max_function_evaluations': 1500 * 100,
               'n_individuals': 100,
               'seed_rng': 0,  # undefined in the original paper
               }
    jade = JADE(problem, options)
    results = jade.optimize()
    print(results)
    print(results['best_so_far_y'])
    # generation 1500
    #            9.4e-66
    # vs         1.3e-54 (from the original paper)

    problem = {'fitness_function': step,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100*np.ones((ndim_problem,)),
               'upper_boundary': 100*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 1500 * 100,
               'n_individuals': 100,
               'seed_rng': 0,  # undefined in the original paper
               }
    jade = JADE(problem, options)
    results = jade.optimize()
    print(results)
    print(results['best_so_far_y'])
    # generation 1500
    #            0
    # vs         0 (from the original paper)

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -30*np.ones((ndim_problem,)),
               'upper_boundary': 30*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 3000 * 100,
               'n_individuals': 100,
               'seed_rng': 0,  # undefined in the original paper
               }
    jade = JADE(problem, options)
    results = jade.optimize()
    print(results)
    print(results['best_so_far_y'])
    # generation 3000
    #            0.0
    # vs         3.2e-01 (from the original paper)

    problem = {'fitness_function': rastrigin,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5.12*np.ones((ndim_problem,)),
               'upper_boundary': 5.12*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 5000 * 100,
               'n_individuals': 100,
               'seed_rng': 0,  # undefined in the original paper
               }
    jade = JADE(problem, options)
    results = jade.optimize()
    print(results)
    print(results['best_so_far_y'])
    # generation 5000
    #            0
    # vs         0 (from the original paper)

    problem = {'fitness_function': ackley,
               'ndim_problem': ndim_problem,
               'lower_boundary': -32*np.ones((ndim_problem,)),
               'upper_boundary': 32*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 2000 * 100,
               'n_individuals': 100,
               'seed_rng': 0,  # undefined in the original paper
               }
    jade = JADE(problem, options)
    results = jade.optimize()
    print(results)
    print(results['best_so_far_y'])
    # generation 2000
    #            4.0e-15
    # vs         4.4e-15 (from the original paper)
