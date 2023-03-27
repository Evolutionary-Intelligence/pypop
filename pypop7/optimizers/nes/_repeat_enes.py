"""Repeat the following paper for `ENES`:
    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014.
    Natural evolution strategies.
    Journal of Machine Learning Research, 15(1), pp.949-980.
    https://jmlr.org/papers/v15/wierstra14a.html
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/nes.py



    import numpy as np
    from pybrain.optimization.distributionbased.nes import ExactNES as ENES


    def sphere(x):  # to be maximized
        return -np.sum(np.power(x, 2))


    np.random.seed(5)
    solver = ENES(sphere, 4*np.ones((10,)), maxEvaluations=5e5, verbose=True, importanceMixing=False)
    solver.x = 4*np.ones((10,))
    solver.learn()
    # ('Evals:', 43100)
    # ('Step:', 430, 'best:', -2.2205949442220285e-20)
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import sphere
from pypop7.optimizers.nes.enes import ENES as Solver


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 10
    for f in [sphere]:
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem}
        options = {'max_function_evaluations': 43100,
                   'seed_rng': 0,
                   'x': 4*np.ones((ndim_problem,)),
                   'saving_fitness': 1,
                   'is_restart': False}
        solver = Solver(problem, options)
        results = solver.optimize()
        print(results)
        print(results['best_so_far_y'])  # 2.295365297463382e-20
        print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
