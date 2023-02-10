"""Repeat the following paper for `ONES`:
    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014.
    Natural evolution strategies.
    Journal of Machine Learning Research, 15(1), pp.949-980.
    https://jmlr.org/papers/v15/wierstra14a.html
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/nes.py



    import numpy as np
    from pybrain.optimization.distributionbased.nes import OriginalNES as ONES


    def sphere(x):
        return -np.sum(np.power(x, 2))


    np.random.seed(5)
    solver = ONES(sphere, 4*np.ones((10,)), maxEvaluations=5e5, verbose=True, importanceMixing=False)
    solver.x = 4*np.ones((10,))
    solver.learn()
    # ('Evals:', 28200)
    # ('Step:', 281, 'best:', -5.104246707867472e-20)
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import sphere
from pypop7.optimizers.nes.ones import ONES as Solver


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 10
    for f in [sphere]:
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem}
        options = {'max_function_evaluations': 28200,
                   'seed_rng': 4,
                   'x': 4 * np.ones((ndim_problem,)),
                   'saving_fitness': 1,
                   'is_restart': False}
        solver = Solver(problem, options)
        results = solver.optimize()
        print(results)
        print(results['best_so_far_y'])
        # seed_rng | best_so_far_y
        # 0          1.21231855e-19
        # 1          9.55072179e-20
        # 2          2.02843481e-19
        # 3          1.38791180e-19
        # 4          5.10282137e-20
        print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
