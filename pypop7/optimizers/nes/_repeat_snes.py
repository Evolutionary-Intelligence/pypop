"""Repeat the following paper for `SNES`:
    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014.
    Natural evolution strategies.
    Journal of Machine Learning Research, 15(1), pp.949-980.
    https://jmlr.org/papers/v15/wierstra14a.html
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/snes.py



    import numpy as np
    from pybrain.optimization.distributionbased.snes import SNES


    def cigar(x):  # for maximum
        x = np.power(x, 2)
        y = x[0] + (10 ** 6) * np.sum(x[1:])
        return -y


    np.random.seed(5)
    solver = SNES(cigar, 4*np.ones((1000,)), maxEvaluations=5e5, verbose=True)
    solver.x = 4*np.ones((1000,))
    solver.learn()
    # ('Step:', 20832, 'best:', 0.000598354700108173)  # for minimum
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import cigar
from pypop7.optimizers.nes.snes import SNES as Solver


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 1000
    for f in [cigar]:
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem}
        options = {'max_function_evaluations': 5e5,
                   'seed_rng': 0,
                   'x': 4*np.ones((ndim_problem,)),
                   'sigma': 1.0,
                   'saving_fitness': 1,
                   'is_restart': False}
        solver = Solver(problem, options)
        results = solver.optimize()
        print(results)
        print(results['best_so_far_y'])  # 0.0005645391876666976
        print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
