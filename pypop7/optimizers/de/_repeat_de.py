"""Repeat DifferentialEvolution algorithm in Scipy library
    The code is as following:
    import numpy as np
    import time
    from scipy.optimize import differential_evolution

    def sphere(x):
        y = np.sum(np.power(x, 2))
        return y

    start_time = time.time()
    ndim_problem = 30
    result = differential_evolution(sphere, bounds=[(-100, 100)] * ndim_problem, strategy='best1bin',
                                popsize=100, maxiter=800, disp=True)
    print("result x:", result.x, "result f:", result.fun)
    print("Runtime: {:7.5e}".format(time.time() - start_time))

    Compare pypop's JADE algorithm with it:
    scipy result:
    best: 2.11193e-24

    pypop result:
    best: 2.1046682114162533e-31
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere
from pypop7.optimizers.de.jade import JADE

if __name__ == '__main__':
    ndim_problem = 30
    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem,
               'lower_boundary': -100 * np.ones((ndim_problem,)),
               'upper_boundary': 100 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 800 * 100,
               'n_individuals': 100,
               'seed_rng': 0,  # undefined in the original paper
               }
    jade = JADE(problem, options)
    results = jade.optimize()
    print(results)
    print(results['best_so_far_y'])
