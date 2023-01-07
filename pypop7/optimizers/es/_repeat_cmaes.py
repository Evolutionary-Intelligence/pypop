"""Repeat the following paper for `CMAES`:
    Hansen, N., 2016.
    The CMA evolution strategy: A tutorial.
    arXiv preprint arXiv:1604.00772.
    https://arxiv.org/abs/1604.00772
    https://github.com/CyberAgentAILab/cmaes

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
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import ellipsoid, sphere
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
    #                   [3.00000000e+03, 2.29627056e+05],
    #                   [6.00000000e+03, 3.98863980e+04],
    #                   [9.00000000e+03, 5.81068731e+03],
    #                   [1.20000000e+04, 2.78377777e+03],
    #                   [1.50000000e+04, 1.46719576e+03],
    #                   [1.80000000e+04, 5.58938266e+02],
    #                   [2.10000000e+04, 1.50634692e+02],
    #                   [2.40000000e+04, 8.28571949e+01],
    #                   [2.70000000e+04, 5.27908745e+01],
    #                   [3.00000000e+04, 1.78872473e+01],
    #                   [3.30000000e+04, 5.11224718e+00],
    #                   [3.60000000e+04, 6.67062161e-01],
    #                   [3.90000000e+04, 2.00268740e-02],
    #                   [4.20000000e+04, 2.48809390e-04],
    #                   [4.50000000e+04, 3.93840831e-06],
    #                   [4.80000000e+04, 8.50976589e-08],
    #                   [5.10000000e+04, 5.23197175e-13]])
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
    #                   [3.00000000e+03, 2.81648743e-04],
    #                   [6.00000000e+03, 3.77063006e-10]])
