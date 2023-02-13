"""Repeat the following paper for `COEA`:
    Potter, M.A. and De Jong, K.A., 1994, October.
    A cooperative coevolutionary approach to function optimization.
    In International Conference on Parallel Problem Solving from Nature (pp. 249-257).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/3-540-58484-6_269

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/coea

    The original paper used the binary representation, which is *rarely* used for continuous optimization.
    Our implementation uses the real-valued representation, with a different sub-optimizer.

    Luckily our code could still repeat the data reported in the original paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import rastrigin, griewank, ackley
from pypop7.optimizers.cc.coea import COEA as Solver


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')

    ndim_problem = 10
    problem = {'fitness_function': griewank,
               'ndim_problem': ndim_problem,
               'lower_boundary': -600 * np.ones((ndim_problem,)),
               'upper_boundary': 600 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 100000,
               'seed_rng': 0,
               'saving_fitness': 1}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    fitness = results['fitness']
    plt.plot(fitness[:, 0], fitness[:, 1], 'k')
    plt.xticks([0, 20000, 40000, 60000, 80000, 100000])
    plt.xlim([0, 100000])
    plt.xlabel('function evaluations')
    plt.yticks([0, 2, 4, 6, 8])
    plt.ylim([0, 8])
    plt.ylabel('best individual')
    plt.show()

    ndim_problem = 20
    problem = {'fitness_function': rastrigin,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5 * np.ones((ndim_problem,)),
               'upper_boundary': 5 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 100000,
               'seed_rng': 0,
               'saving_fitness': 1}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    fitness = results['fitness']
    plt.plot(fitness[:, 0], fitness[:, 1], 'k')
    plt.xticks([0, 20000, 40000, 60000, 80000, 100000])
    plt.xlim([0, 100000])
    plt.xlabel('function evaluations')
    plt.yticks([0, 10, 20, 30, 40])
    plt.ylim([0, 40])
    plt.ylabel('best individual')
    plt.show()

    ndim_problem = 30
    problem = {'fitness_function': ackley,
               'ndim_problem': ndim_problem,
               'lower_boundary': -600 * np.ones((ndim_problem,)),
               'upper_boundary': 600 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 100000,
               'seed_rng': 0,
               'saving_fitness': 1}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    fitness = results['fitness']
    plt.plot(fitness[:, 0], fitness[:, 1], 'k')
    plt.xticks([0, 20000, 40000, 60000, 80000, 100000])
    plt.xlim([0, 100000])
    plt.xlabel('function evaluations')
    plt.yticks([0, 4, 8, 12, 16])
    plt.ylim([0, 16])
    plt.ylabel('best individual')
    plt.show()
