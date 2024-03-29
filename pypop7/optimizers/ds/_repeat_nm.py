"""Repeat the following paper for `NM`:
    Nelder, J.A. and Mead, R., 1965.
    A simplex method for function minimization.
    The Computer Journal, 7(4), pp.308-313.
    https://academic.oup.com/comjnl/article-abstract/7/4/308/354237

    Luckily our Python code could repeat the data generated by the other Python code *well*.
    Therefore, we argue that its repeatability could be **well-documented**.



    The Python reference script is given below (note that first install `pymoo` via `pip install pymoo`):
    -----------------------------------------------------------------------------------------------------
    from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
    from pymoo.problems.single import Ackley
    from pymoo.optimize import minimize

    problem = Ackley(n_var=100)
    algorithm = NelderMead(init_delta=0.1)
    res = minimize(problem=problem, algorithm=algorithm, termination=('n_eval', 1e5), verbose=True, seed=0)
    print(res)
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import ackley
from pypop7.optimizers.ds.nm import NM as Solver


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 100
    for f in [ackley]:
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -32.768*np.ones((ndim_problem,)),
                   'upper_boundary': 32.768*np.ones((ndim_problem,))}
        options = {'max_function_evaluations': 1e5,
                   'seed_rng': 0,
                   'sigma': 0.1,
                   'is_restart': False}
        solver = Solver(problem, options)
        results = solver.optimize()
        print(results)  # 19.805781729365254 vs 1.958869E+01 (from pymoo)
        print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
