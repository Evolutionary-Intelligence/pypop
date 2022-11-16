"""Repeat the following paper for `IPSO`:
    De Oca, M.A.M., Stutzle, T., Van den Enden, K. and Dorigo, M., 2010.
    Incremental social learning in particle swarms.
    IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 41(2), pp.368-384.
    https://ieeexplore.ieee.org/document/5582312
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import ackley
from pypop7.optimizers.pso.ipso import IPSO as Solver

if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 100
    for f in [ackley]:
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -32.768 * np.ones((ndim_problem,)),
                   'upper_boundary': 32.768 * np.ones((ndim_problem,))}
        options = {'max_function_evaluations': 1e6,
                   'verbose': 1e3}
        solver = Solver(problem, options)
        results = solver.optimize()
        print(results)      # 4.093170247188027e-12 vs 0.19331169 (from pymoo)
        print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
