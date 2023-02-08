"""Repeat the following paper for `SGES`:
    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014.
    Natural evolution strategies.
    Journal of Machine Learning Research, 15(1), pp.949-980.
    https://jmlr.org/papers/v15/wierstra14a.html
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/ves.py



    import time
    import numpy as np
    from pybrain.optimization.distributionbased.ves import VanillaGradientEvolutionStrategies as VGES


    def sphere(x):
        return np.sum(np.power(x, 2))


    solver = VGES(sphere, 4*np.ones((10,)), minimize=True, maxEvaluations=2e6, verbose=True, importanceMixing=False)
    start_time = time.time()
    solver.learn()
    print("Runtime: {:7.5e}".format(time.time() - start_time))
    # Numerical Instability. Stopping.
    # Evals: 0 Step: 0 best: None
    # (array([4., 4., 4., 4., 4., 4., 4., 4., 4., 4.]), None)
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import sphere
from pypop7.optimizers.nes.sges import SGES as Solver


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 10
    for f in [sphere]:
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem}
        options = {'max_function_evaluations': 100000 * ndim_problem,
                   'seed_rng': 0,
                   'x': 4 * np.ones((ndim_problem,)),
                   'n_individuals': 50,
                   'lr_sigma': 0.02,
                   'saving_fitness': 1,
                   'is_restart': False}
        solver = Solver(problem, options)
        results = solver.optimize()
        print(results)
        print(results['best_so_far_y'])  # 51.532660164536765
        print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
