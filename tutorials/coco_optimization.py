"""This is a simple demo for `PyPop7` optimization on the well-designed `COCO` platform.

    If you cannot install `cocoex` on Windows, the solution is given in:
        https://github.com/numbbo/coco/issues/2086

    Note that this script is only used to check whether or not `cocoex` is installed
    *successfully*. For benchmarking on `cocoex`, please run the following code:
    https://github.com/Evolutionary-Intelligence/pypop/blob/main/tutorials/coco_benchmarking.py

    References
    ----------
    N. Hansen, A. Auger, R. Ros, O. Mersmann, T. Tušar, D. Brockhoff.
    COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting.
    Optimization Methods and Software, 36(1), pp.114-144, 2021.
"""
import cocoex
import numpy as np

from pypop7.optimizers.ds.nm import NM as Solver


if __name__ == '__main__':
    print(cocoex.known_suite_names)
    suite = cocoex.Suite('bbob', '', '')
    for current_problem in suite:
        print(current_problem)
        d = current_problem.dimension
        problem = {'fitness_function': current_problem,
                   'ndim_problem': d,
                   'lower_boundary': -10 * np.ones((d,)),
                   'upper_boundary': 10 * np.ones((d,))}
        options = {'max_function_evaluations': 1e3 * d,
                   'seed_rng': 2022,
                   'sigma': 1.0,
                   'verbose': False,
                   'saving_fitness': 2000}
        solver = Solver(problem, options)
        results = solver.optimize()
        print('  best-so-far fitness:', results['best_so_far_y'])
