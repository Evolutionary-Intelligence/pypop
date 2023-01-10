"""Repeat the following paper for `DDCMA`:
    Akimoto, Y. and Hansen, N., 2020.
    Diagonal acceleration for covariance matrix adaptation evolution strategies.
    Evolutionary Computation, 28(3), pp.405-435.
    https://direct.mit.edu/evco/article/28/3/405/94999/Diagonal-Acceleration-for-Covariance-Matrix

    Luckily our Python code could repeat the data reported in the following Python code *well*.
    Therefore, we argue that its repeatability could be **well-documented**.

    The following Python code is based on https://gist.github.com/youheiakimoto/1180b67b5a0b1265c204cba991fa8518:
    -------------------------------------------------------------------------------------------------------------
    N = 40


    def fobj(x):  # ellipsoid
        y = np.empty((x.shape[0],))
        for j in range(x.shape[0]):
            y[j] = sum([(1000 ** (i / (x.shape[1] - 1)) * x[j, i]) ** 2 for i in range(x.shape[1])])
        return y


    total_neval = 0   # total number of f-calls
    F_TARGET = 9.05738422e-92
    # Main loop
    ddcma = DdCma(xmean0=3 * np.ones(N), sigma0=2.0*np.ones(N))
    logger = Logger(ddcma)
    issatisfied = False
    fbestsofar = np.inf
    while not issatisfied:
        ddcma.onestep(func=fobj)
        fbest = np.min(ddcma.arf)
        fbestsofar = min(fbest, fbestsofar)
        if fbest <= F_TARGET:
            issatisfied = True
        if ddcma.t % 200 == 0:
            print(ddcma.t, ddcma.neval, fbest, fbestsofar)


    200 3000 1668.9263013816555 1595.6819077277642
    400 6000 3.0776542517887293 2.446659749508933
    600 9000 2.0835078646085184e-06 1.8057099297427967e-06
    800 12000 3.820665797855719e-12 3.820665797855719e-12
    1000 15000 2.716880694923014e-18 2.453420258106028e-18
    1200 18000 1.2258980046389397e-24 1.052532629550606e-24
    1400 21000 1.682190757968126e-30 1.682190757968126e-30
    1600 24000 1.704443002486423e-36 1.704443002486423e-36
    1800 27000 1.9146782471495856e-42 1.7610828288522217e-42
    2000 30000 2.4894475408934418e-48 2.4894475408934418e-48
    2200 33000 7.224012538213152e-55 7.224012538213152e-55
    2400 36000 1.9863861444356578e-60 1.486573123176647e-60
    2600 39000 1.985436013018233e-66 1.985436013018233e-66
    2800 42000 1.1196881877790812e-72 1.1196881877790812e-72
    3000 45000 6.961990048960852e-79 6.961990048960852e-79
    3200 48000 4.973876492330526e-85 4.973876492330526e-85
    3400 51000 2.4077717597886575e-91 1.9969365655801173e-91
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import ellipsoid, sphere, rosenbrock, step
from pypop7.optimizers.es.ddcma import DDCMA as Solver


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 40
    problem = {'fitness_function': ellipsoid,
               'ndim_problem': ndim_problem}
    options = {'max_function_evaluations': 51000,
               'seed_rng': 0,
               'x': 3 * np.ones((ndim_problem,)),  # mean
               'sigma': 2.0,
               'is_restart': False,
               'saving_fitness': 3000}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
    # 'fitness': array([[1.00000000e+00, 5.14940030e+07],
    #                   [3.00000000e+03, 1.16701163e+03],
    #                   [6.00000000e+03, 2.55921669e-01],
    #                   [9.00000000e+03, 6.03649703e-07],
    #                   [1.20000000e+04, 5.92998046e-13],
    #                   [1.50000000e+04, 7.99457850e-19],
    #                   [1.80000000e+04, 1.01696489e-24],
    #                   [2.10000000e+04, 1.14664675e-30],
    #                   [2.40000000e+04, 2.16305533e-36],
    #                   [2.70000000e+04, 3.51879595e-42],
    #                   [3.00000000e+04, 2.40894019e-48],
    #                   [3.30000000e+04, 1.49335827e-54],
    #                   [3.60000000e+04, 1.75442657e-60],
    #                   [3.90000000e+04, 7.69981804e-67],
    #                   [4.20000000e+04, 7.84220954e-73],
    #                   [4.50000000e+04, 3.71117826e-79],
    #                   [4.80000000e+04, 1.91388887e-85],
    #                   [5.10000000e+04, 9.05738422e-92]]
    start_run = time.time()
    ndim_problem = 40
    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem}
    options = {'max_function_evaluations': 6000,
               'seed_rng': 0,
               'x': 3 * np.ones((ndim_problem,)),  # mean
               'sigma': 2.0,
               'is_restart': False,
               'saving_fitness': 3000}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
    # 'fitness': array([[1.00000000e+00, 7.65494377e+02],
    #                   [3.00000000e+03, 2.05166721e-04],
    #                   [6.00000000e+03, 2.89571316e-10]]
    start_run = time.time()
    ndim_problem = 40
    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem}
    options = {'max_function_evaluations': 40000,
               'seed_rng': 0,
               'x': 3 * np.ones((ndim_problem,)),  # mean
               'sigma': 2.0,
               'is_restart': False,
               'saving_fitness': 3000}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
    # 'fitness': array([[1.00000000e+00, 3.06775056e+06],
    #                   [3.00000000e+03, 3.94528616e+01],
    #                   [6.00000000e+03, 3.92399627e+01],
    #                   [9.00000000e+03, 3.76547255e+01],
    #                   [1.20000000e+04, 3.52135237e+01],
    #                   [1.50000000e+04, 3.25782538e+01],
    #                   [1.80000000e+04, 3.02754946e+01],
    #                   [2.10000000e+04, 2.74298979e+01],
    #                   [2.40000000e+04, 2.46997894e+01],
    #                   [2.70000000e+04, 2.20156904e+01],
    #                   [3.00000000e+04, 1.91551581e+01],
    #                   [3.30000000e+04, 1.65238778e+01],
    #                   [3.60000000e+04, 1.38376394e+01],
    #                   [3.90000000e+04, 1.11631159e+01],
    #                   [4.00000000e+04, 1.01679117e+01]]
    start_run = time.time()
    ndim_problem = 40
    problem = {'fitness_function': step,
               'ndim_problem': ndim_problem}
    options = {'max_function_evaluations': 3000,
               'seed_rng': 0,
               'x': 3 * np.ones((ndim_problem,)),  # mean
               'sigma': 2.0,
               'saving_fitness': 300,
               'is_restart': False}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
    # 'fitness': array([[1.0e+00, 7.7e+02],
    #                   [3.00e+02, 1.16e+02],
    #                   [6.00e+02, 2.00e+01],
    #                   [9.00e+02, 6.00e+00],
    #                   [1.20e+03, 3.00e+00],
    #                   [1.50e+03, 1.00e+00],
    #                   [1.80e+03, 1.00e+00],
    #                   [2.10e+03, 1.00e+00],
    #                   [2.40e+03, 1.00e+00],
    #                   [2.70e+03, 1.00e+00],
    #                   [3.00e+03, 1.00e+00]]
