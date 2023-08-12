"""Repeat the following paper for `R1NES`:
    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014.
    Natural evolution strategies.
    Journal of Machine Learning Research, 15(1), pp.949-980.
    https://jmlr.org/papers/v15/wierstra14a.html
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/rank1.py



    import numpy as np
    from pybrain.optimization.distributionbased.rank1 import Rank1NES


    def cigar(x):  # maximization
        x = np.power(x, 2)
        y = x[0] + (10 ** 6) * np.sum(x[1:])
        return -y


    np.random.seed(5)
    solver = Rank1NES(cigar, 4*np.ones((100,)), maxEvaluations=5e5, verbose=True)
    solver.x = 4*np.ones((100,))
    solver.learn()  # -3.4497462517488313e-06 (maximization)
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import cigar
from pypop7.optimizers.nes.r1nes import R1NES as Solver


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 100
    for f in [cigar]:
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem}
        options = {'max_function_evaluations': 5e5,
                   'seed_rng': 2,
                   'x': 4*np.ones((ndim_problem,)),
                   'saving_fitness': 1,
                   'is_restart': False}
        solver = Solver(problem, options)
        results = solver.optimize()
        print(results)
        print(results['best_so_far_y'])  # 1.925873624433685e-06 (minimization)
        print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
