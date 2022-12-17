"""Repeat the following paper for `RHC`:
    Schaul, T., Bayer, J., Wierstra, D., Sun, Y., Felder, M., Sehnke, F., Rückstieß, T. and Schmidhuber, J., 2010.
    PyBrain.
    Journal of Machine Learning Research, 11(24), pp.743-746.
    https://jmlr.org/papers/v11/schaul10a.html

    Luckily our Python code could repeat the data reported in the other reference Python code *well*.
    Therefore, we argue that the repeatability of `RHC` could be **well-documented**.

    The reference Python code (https://github.com/pybrain/pybrain) is shown below:
    ------------------------------------------------------------------------------
    import time

    import numpy as np
    # use Python 2.7 to avoid possible unsuccessful installation, since PyBrain was not maintained (after 2017)
    from pybrain.optimization.hillclimber import HillClimber as RHC


    def ellipsoid(x):
        x = np.power(x, 2)
        y = np.dot(np.power(10, 6 * np.linspace(0, 1, x.size)), x)
        return y


    solver = RHC(ellipsoid, 4 * np.ones((1000,)), minimize=True, maxEvaluations=2e6, verbose=True)
    start_time = time.time()
    solver.learn()
    # Step: 1999998 best: 11499639.24811204 (note that different runs result in slightly different results)
    print("Runtime: {:7.5e}".format(time.time() - start_time))
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import ellipsoid
from pypop7.optimizers.rs.rhc import RHC


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 1000
    problem = {'fitness_function': ellipsoid,
               'ndim_problem': ndim_problem,
               'upper_boundary': 5.0 * np.ones((ndim_problem,)),
               'lower_boundary': -5.0 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 2e6,
               'seed_rng': 0,
               'x': 4 * np.ones((ndim_problem,)),
               'sigma': 0.1,
               'verbose': 200000,
               'saving_fitness': 200000}
    print(RHC(problem, options).optimize())  # 1.06803861e+07
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
