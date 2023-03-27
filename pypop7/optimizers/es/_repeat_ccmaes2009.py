"""Repeat the following paper for `CCMAES2009`:
    Suttorp, T., Hansen, N. and Igel, C., 2009.
    Efficient covariance matrix update for variable metric evolution strategies.
    Machine Learning, 75(2), pp.167-197.
    https://link.springer.com/article/10.1007/s10994-009-5102-1

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/ccmaes2009

    Luckily our Python code could repeat the data reported in the paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.optimizers.es.ccmaes2009 import CCMAES2009 as Solver
from pypop7.benchmarks.base_functions import ellipsoid, rosenbrock


def plot(function, ndim):
    plt.figure()
    res = pickle.load(open(function + '_' + str(ndim) + '.pickle', 'rb'))
    plt.yscale('log')
    if function == 'ellipsoid':
        plt.ylim([1e-35, 1e5])
        plt.yticks([1e-35, 1e-30, 1e-25, 1e-20, 1e-15, 1e-10, 1e-5, 1e0, 1e5])
        plt.xlim([0, 3e4])
        plt.xticks([0, 5e3, 1e4, 1.5e4, 2e4, 2.5e4, 3e4])
    elif function == 'rosenbrock':
        plt.ylim([1e-20, 1e5])
        plt.yticks([1e-20, 1e-15, 1e-10, 1e-5, 1e0, 1e5])
        plt.xlim([0, 3e4])
        plt.xticks([0, 5e3, 1e4, 1.5e4, 2e4, 2.5e4, 3e4])
    plt.plot(res['fitness'][:, 0], res['fitness'][:, 1], color='r')
    plt.xlabel('objective function evaluations')
    plt.ylabel('objective function value')
    plt.title(function.capitalize())
    plt.show()


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    for f in [ellipsoid, rosenbrock]:
        for d in [20]:
            print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
            problem = {'fitness_function': f,
                       'ndim_problem': d,
                       'lower_boundary': 0.1*np.ones((d,)),
                       'upper_boundary': 0.3*np.ones((d,))}
            options = {'fitness_threshold': 1e-10,
                       'seed_rng': 0,
                       'sigma': 0.2/3,
                       'saving_fitness': 1,
                       'is_restart': False}
            if f.__name__ == 'ellipsoid':
                options['fitness_threshold'] = 1e-35
                options['max_function_evaluations'] = 3e4
            elif f.__name__ == 'rosenbrock':
                options['fitness_threshold'] = 1e-20
                options['max_function_evaluations'] = 3e4
            solver = Solver(problem, options)
            results = solver.optimize()
            pickle.dump(results, open(f.__name__ + '_' + str(d) + '.pickle', 'wb'))
            plot(f.__name__, d)
