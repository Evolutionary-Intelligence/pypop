"""Repeat StochasticHillClimber algorithm in pybrain with python version=2.7
    The code is as following:
    import time
    import numpy as np
    from pybrain.optimization.hillclimber import StochasticHillClimber as SHC

    def ellipsoid(x):
        x = np.power(x, 2)
        y = np.dot(np.power(10, 6 * np.linspace(0, 1, x.size)), x)
        return y

    solver = SHC(ellipsoid, 4 * np.ones((1000,)), minimize=True, maxEvaluations=2e6, verbose=True)
    start_time = time.time()
    solver.learn()
    print("Runtime: {:7.5e}".format(time.time() - start_time))

    pybrain result:
    best: 12968919.810064487

    pypop result:
    best: 10389166.423976313
"""
import numpy as np
import time

from pypop7.benchmarks.base_functions import ellipsoid
from pypop7.optimizers.rs.arhc import ARHC


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 1000
    problem = {'fitness_function': ellipsoid,
               'ndim_problem': ndim_problem,
               'upper_boundary': 5.0 * np.ones((ndim_problem,)),
               'lower_boundary': -5.0 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 2e6,
               'fitness_threshold': 1e-10,
               'seed_rng': 0,
               'x': 4 * np.ones((ndim_problem,)),
               'sigma': 0.1,
               'temperature': 100,
               'verbose': 200000,
               'saving_fitness': 200000}
    arhc = ARHC(problem, options)
    results = arhc.optimize()
    print(results)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))