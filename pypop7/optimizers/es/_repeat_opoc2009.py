"""Repeat the following paper for `OPOC2009`:
    Suttorp, T., Hansen, N. and Igel, C., 2009.
    Efficient covariance matrix update for variable metric evolution strategies.
    Machine Learning, 75(2), pp.167-197.
    https://link.springer.com/article/10.1007/s10994-009-5102-1

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/opoc2009

    Luckily our Python code could repeat the data reported in the paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import ellipsoid, cigar
from pypop7.optimizers.es.opoc2009 import OPOC2009 as Solver


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    # plot fig.4
    d = 20
    problem = {'fitness_function': ellipsoid,
               'ndim_problem': d,
               'lower_boundary': 0.1*np.ones((d,)),
               'upper_boundary': 0.3*np.ones((d,))}
    options = {'fitness_threshold': 1e-250,
               'max_function_evaluations': 6e4,
               'seed_rng': 0,
               'sigma': 0.3,  # not given in the original paper
               'saving_fitness': 1,
               'is_restart': False}
    solver = Solver(problem, options)
    results = solver.optimize()
    pickle.dump(results, open(ellipsoid.__name__ + '_' + str(d) + '.pickle', 'wb'))
    plt.figure()
    result = pickle.load(open(ellipsoid.__name__ + '_' + str(d) + '.pickle', 'rb'))
    plt.xlim([0, 1e5])
    plt.xticks([0, 2e4, 4e4, 6e4, 8e4, 1e5])
    plt.xlabel("objective function evaluations")
    plt.yscale('log')
    plt.ylim([1e-250, 1e50])
    plt.yticks([1e-250, 1e-200, 1e-150, 1e-100, 1e-50, 1e0, 1e50])
    plt.ylabel("objective function value")
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='green')
    plt.title(ellipsoid.__name__.capitalize())
    plt.show()
    # plot fig.5
    evaluations = []
    dims = [5, 10, 20, 40, 80, 160, 320]
    for d in dims:
        problem = {'fitness_function': cigar,
                   'ndim_problem': d,
                   'lower_boundary': 0.1*np.ones((d,)),
                   'upper_boundary': 0.3*np.ones((d,))}
        options = {'fitness_threshold': 1e-15,
                   'max_function_evaluations': 1e6,
                   'seed_rng': 0,
                   'sigma': 0.3,  # not given in the original paper
                   'saving_fitness': 1,
                   'is_restart': False}
        solver = Solver(problem, options)
        results = solver.optimize()
        evaluations.append(results['n_function_evaluations'])
    plt.figure()
    plt.xscale('log')
    plt.xlim([1, 5e2])
    plt.xticks([10, 100])
    plt.xlabel('search space dimension n')
    plt.yscale('log')
    plt.ylim([1e3, 1e7])
    plt.yticks([1e3, 1e4, 1e5, 1e6, 1e7])
    plt.ylabel('objective function evaluations')
    plt.plot(dims, evaluations, 'o--', color='green')
    plt.title(cigar.__name__.capitalize())
    plt.show()
