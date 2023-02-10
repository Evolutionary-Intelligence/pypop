"""Repeat the following paper for `SGES`:
    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014.
    Natural evolution strategies.
    Journal of Machine Learning Research, 15(1), pp.949-980.
    https://jmlr.org/papers/v15/wierstra14a.html
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/ves.py



    import numpy as np
    from pybrain.optimization.distributionbased.ves import VanillaGradientEvolutionStrategies as VGES


    def sphere(x):  # for maximization
        return -np.sum(np.power(x, 2))


    np.random.seed(5)
    solver = VGES(sphere, 4*np.ones((10,)), maxEvaluations=5e5, verbose=True, importanceMixing=False)
    solver.x = 4*np.ones((10,))
    solver.learn()
    # ('Evals:', 500000)
    # ('Step:', 4999, 'best:', -2.375081667889928)
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
        options = {'max_function_evaluations': 5e5,
                   'seed_rng': 1,
                   'x': 4 * np.ones((ndim_problem,)),
                   'saving_fitness': 1,
                   'is_restart': False}
        solver = Solver(problem, options)
        results = solver.optimize()
        print(results)
        print(results['best_so_far_y'])
        # seed_rng | best_so_far_y
        # 0          4.84043450e+00
        # 1          2.52480970e+00
        # 2          9.66536022e+00
        # 3          2.56613399e+00
        # 4          1.04503107e+01
        print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
