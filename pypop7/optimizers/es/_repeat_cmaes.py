"""Repeat the following paper for `CMAES`:
    Hansen, N., 2016.
    The CMA evolution strategy: A tutorial.
    arXiv preprint arXiv:1604.00772.
    https://arxiv.org/abs/1604.00772
    https://github.com/CyberAgentAILab/cmaes

    Luckily our Python code could repeat the data reported in the following Python code *well*.
    Therefore, we argue that its repeatability could be **well-documented**.

    The following Python code is based on https://github.com/CyberAgentAILab/cmaes:
    -------------------------------------------------------------------------------
    import numpy as np
    from cmaes import CMA


    def ellipsoid(x):
        n = len(x)
        return sum([(1000 ** (i / (n - 1)) * x[i]) ** 2 for i in range(n)])


    dim = 40
    optimizer = CMA(mean=3 * np.ones(dim), sigma=2.0, seed=0)
    evals = 0
    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = ellipsoid(x)
            evals += 1
            solutions.append((x, value))
            if evals % 3000 == 0:
                print(f"{evals:5d}  {value:10.5f}")
        optimizer.tell(solutions)


     3000  179337.29807
     6000  16318.30196
     9000  3089.93275
    12000  1697.10943
    15000   717.80845
    18000   254.04115
    21000    97.67904
    24000    27.28688
    27000     8.10173
    30000     3.83081
    33000     1.16855
    36000     0.29288
    39000     0.04379
    42000     0.01118
    45000     0.00037
    48000     0.00000
    51000     0.00000


    def sphere(x):
        return np.sum(np.power(x, 2))


    3000     0.00051
    6000     0.00000


    def rosenbrock(x):
        return 100 * np.sum(np.power(x[1:] - np.power(x[:-1], 2), 2)) + np.sum(np.power(x[:-1] - 1, 2))


     3000   325.04031
     6000    47.97352
     9000    32.97643
    12000    30.34539
    15000    27.74735
    18000    25.45727
    21000    23.47825
    24000    21.35947
    27000    19.31040
    30000    16.95758
    33000    14.92188
    36000    12.75626
    39000    10.15624
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import ellipsoid, sphere, rosenbrock
from pypop7.optimizers.es.cmaes import CMAES as Solver


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 40
    problem = {'fitness_function': ellipsoid,
               'ndim_problem': ndim_problem}
    options = {'max_function_evaluations': 51000,
               'seed_rng': 0,
               'x': 3 * np.ones((ndim_problem,)),  # mean
               'sigma': 2.0,
               'saving_fitness': 3000}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
    # 'fitness': array([[1.00000000e+00, 5.14940030e+07],
    #                   [3.00000000e+03, 9.35191308e+04],
    #                   [6.00000000e+03, 1.37736660e+04],
    #                   [9.00000000e+03, 2.64808281e+03],
    #                   [1.20000000e+04, 1.17517466e+03],
    #                   [1.50000000e+04, 5.16757676e+02],
    #                   [1.80000000e+04, 2.53007274e+02],
    #                   [2.10000000e+04, 1.93887396e+02],
    #                   [2.40000000e+04, 1.45301836e+02],
    #                   [2.70000000e+04, 8.23838566e+01],
    #                   [3.00000000e+04, 3.17710025e+01],
    #                   [3.30000000e+04, 1.31033420e+01],
    #                   [3.60000000e+04, 1.89760833e+00],
    #                   [3.90000000e+04, 2.92947011e-01],
    #                   [4.20000000e+04, 1.41835727e-02],
    #                   [4.50000000e+04, 7.43703406e-04],
    #                   [4.80000000e+04, 5.51932987e-07],
    #                   [5.10000000e+04, 1.29779238e-12]]
    start_run = time.time()
    ndim_problem = 40
    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem}
    options = {'max_function_evaluations': 6000,
               'seed_rng': 0,
               'x': 3 * np.ones((ndim_problem,)),  # mean
               'sigma': 2.0,
               'saving_fitness': 3000}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
    # 'fitness': array([[1.00000000e+00, 7.65494377e+02],
    #                   [3.00000000e+03, 3.28897696e-04],
    #                   [6.00000000e+03, 2.23674889e-10]]
    start_run = time.time()
    ndim_problem = 40
    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem}
    options = {'max_function_evaluations': 40000,
               'seed_rng': 0,
               'x': 3 * np.ones((ndim_problem,)),  # mean
               'sigma': 2.0,
               'saving_fitness': 3000}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
    # 'fitness': array([[1.00000000e+00, 3.06775056e+06],
    #                   [3.00000000e+03, 3.61427682e+01],
    #                   [6.00000000e+03, 3.33879641e+01],
    #                   [9.00000000e+03, 3.06937858e+01],
    #                   [1.20000000e+04, 2.84362740e+01],
    #                   [1.50000000e+04, 2.62406327e+01],
    #                   [1.80000000e+04, 2.40524849e+01],
    #                   [2.10000000e+04, 2.17632326e+01],
    #                   [2.40000000e+04, 1.96613124e+01],
    #                   [2.70000000e+04, 1.74759791e+01],
    #                   [3.00000000e+04, 1.54036750e+01],
    #                   [3.30000000e+04, 1.32985333e+01],
    #                   [3.60000000e+04, 1.11372868e+01],
    #                   [3.90000000e+04, 9.49352016e+00],
    #                   [4.00000000e+04, 8.81329780e+00]]
