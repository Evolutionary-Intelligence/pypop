"""Repeat the following paper for `CSAES`:
    Ostermeier, A., Gawelczyk, A. and Hansen, N., 1994, October.
    Step-size adaptation based on non-local use of selection information
    In International Conference on Parallel Problem Solving from Nature (pp. 189-198).
    Springer, Berlin, Heidelberg.
    http://link.springer.com/chapter/10.1007/3-540-58484-6_263

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/repeatability/csaes/_repeat_csaes.png

    Luckily our Python code could repeat the data reported in the original paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import schwefel12
from pypop7.optimizers.es.csaes import CSAES


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    plt.figure()
    problem = {'fitness_function': schwefel12,
               'ndim_problem': 20,
               'lower_boundary': -65*np.ones((20,)),
               'upper_boundary': 65*np.ones((20,))}
    options = {'max_function_evaluations': 60000,
               'fitness_threshold': 1e-3,
               'seed_rng': 0,  # undefined in the original paper
               'sigma': 2.0,  # undefined in the original paper
               'n_individuals': 10,
               'saving_fitness': 1,
               'is_restart': False}
    csaes = CSAES(problem, options)
    fitness = csaes.optimize()['fitness']
    plt.plot(fitness[:, 0], fitness[:, 1], 'k')
    plt.xticks([0, 10000, 20000, 30000, 40000, 50000, 60000])
    plt.xlim([0, 60000])
    plt.xlabel('function evaluations')
    plt.yticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    plt.ylim([1e-3, 1e5])
    plt.yscale('log')
    plt.ylabel('best function value')
    plt.show()
