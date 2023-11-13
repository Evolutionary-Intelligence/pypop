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


    def step(x):
        return np.sum(np.power(np.floor(x + 0.5), 2))


      300   167.00000
      600    45.00000
      900     9.00000
     1200     4.00000
     1500     0.00000
     1800     2.00000
     2100     1.00000
     2400     0.00000
     2700     0.00000
     3000     0.00000
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import ellipsoid, sphere, rosenbrock, step
from pypop7.optimizers.es.cmaes import CMAES as Solver


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 40
    problem = {'fitness_function': ellipsoid,
               'ndim_problem': ndim_problem}
    options = {'max_function_evaluations': 51000,
               'seed_rng': 0,
               'x': 3*np.ones((ndim_problem,)),  # mean
               'sigma': 2.0,
               'saving_fitness': 3000,
               'is_restart': False}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
    # 'fitness': array([[1.00000000e+00, 5.14940030e+07],
    #                   [3.00000000e+03, 1.05027942e+05],
    #                   [6.00000000e+03, 1.87830081e+04],
    #                   [9.00000000e+03, 3.96172395e+03],
    #                   [1.20000000e+04, 2.28763989e+03],
    #                   [1.50000000e+04, 1.10335667e+03],
    #                   [1.80000000e+04, 2.80662299e+02],
    #                   [2.10000000e+04, 1.10574040e+02],
    #                   [2.40000000e+04, 6.54210065e+01],
    #                   [2.70000000e+04, 5.34429301e+00],
    #                   [3.00000000e+04, 1.29905411e+00],
    #                   [3.30000000e+04, 5.49803977e-01],
    #                   [3.60000000e+04, 1.88333096e-01],
    #                   [3.90000000e+04, 2.74517989e-02],
    #                   [4.20000000e+04, 3.05062896e-03],
    #                   [4.50000000e+04, 3.38790957e-06],
    #                   [4.80000000e+04, 4.34114445e-11],
    #                   [5.10000000e+04, 6.62069307e-17]])
    start_run = time.time()
    ndim_problem = 40
    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem}
    options = {'max_function_evaluations': 6000,
               'seed_rng': 0,
               'x': 3*np.ones((ndim_problem,)),  # mean
               'sigma': 2.0,
               'saving_fitness': 3000}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
    # 'fitness': array([[1.00000000e+00, 7.65494377e+02],
    #                   [3.00000000e+03, 7.53841053e-04],
    #                   [6.00000000e+03, 4.78791937e-10]])
    start_run = time.time()
    ndim_problem = 40
    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem}
    options = {'max_function_evaluations': 40000,
               'seed_rng': 2,
               'x': 3*np.ones((ndim_problem,)),  # mean
               'sigma': 2.0,
               'saving_fitness': 3000}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
    # 'fitness': array([[1.00000000e+00, 6.40553348e+05],
    #                   [3.00000000e+03, 3.79329466e+01],
    #                   [6.00000000e+03, 3.53495214e+01],
    #                   [9.00000000e+03, 3.25100793e+01],
    #                   [1.20000000e+04, 2.99408054e+01],
    #                   [1.50000000e+04, 2.75302712e+01],
    #                   [1.80000000e+04, 2.50175292e+01],
    #                   [2.10000000e+04, 2.27917169e+01],
    #                   [2.40000000e+04, 2.07077933e+01],
    #                   [2.70000000e+04, 1.86235925e+01],
    #                   [3.00000000e+04, 1.64394147e+01],
    #                   [3.30000000e+04, 1.43852893e+01],
    #                   [3.60000000e+04, 1.21368659e+01],
    #                   [3.90000000e+04, 1.00106507e+01],
    #                   [4.00000000e+04, 9.34172844e+00]])
    start_run = time.time()
    ndim_problem = 40
    problem = {'fitness_function': step,
               'ndim_problem': ndim_problem}
    options = {'max_function_evaluations': 3000,
               'seed_rng': 0,
               'x': 3*np.ones((ndim_problem,)),  # mean
               'sigma': 2.0,
               'saving_fitness': 300,
               'is_restart': False}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
    # 'fitness': array([[1.0e+00, 7.7e+02],
    #                   [3.0e+02, 9.3e+01],
    #                   [6.0e+02, 3.0e+01],
    #                   [9.0e+02, 1.2e+01],
    #                   [1.2e+03, 6.0e+00],
    #                   [1.5e+03, 1.0e+00],
    #                   [1.8e+03, 1.0e+00],
    #                   [2.1e+03, 1.0e+00],
    #                   [2.4e+03, 1.0e+00],
    #                   [2.7e+03, 1.0e+00],
    #                   [3.0e+03, 1.0e+00]])
    #
    # def _set_c_w(self):
    #     return np.minimum(1.0 - self.c_1, self._alpha_cov*(1.0/4.0 + self._mu_eff + 1.0/self._mu_eff - 2.0) /
    #                       (np.square(self.ndim_problem + 2.0) + self._alpha_cov*self._mu_eff/2.0))
    #
    #  removing 1.0/4.0 will result in the same result (0.0).
