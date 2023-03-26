"""Repeat experiments on Fig10(a)(b) from the following paper:
    W. Dong, T. Chen, P. Tino, X. Yao
    Scaling Up Estimation of Distribution Algorithms for Continuous Optimization
    IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 17, NO. 6, DECEMBER 2013
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6461934
"""
import sys
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import schwefel12, schwefel221
from pypop7.optimizers.core import optimizer
from pypop7.optimizers.eda.edamcc import EDAMCC as Solver

sys.modules['optimizer'] = optimizer
sns.set_theme(style='darkgrid')


def read_pickle(function, ndim):
    file = function + '_' + str(ndim) + '.pickle'
    with open(file, 'rb') as handle:
        result = pickle.load(handle)
        return result


def write_pickle(function, ndim, result):
    file = open(function + '_' + str(ndim) + '.pickle', 'wb')
    pickle.dump(result, file)
    file.close()


def plot(function):
    plt.figure()
    result = read_pickle(function, 100)
    plt.yscale('log')
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='r')
    plt.xlim([0, 1e6])
    plt.ylim([1e-30, 1e10])
    plt.xticks([0, 2e5, 4e5, 6e5, 8e5, 1e6])
    plt.yticks([1e-30, 1e-20, 1e-10, 1e0, 1e10])
    plt.xlabel("#evals")
    plt.ylabel("F(x)-F(x*)")
    plt.title(function.capitalize())
    plt.show()


if __name__ == '__main__':
    ndim_problem = 100
    for f in [schwefel221, schwefel12]:
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -10 * np.ones((ndim_problem,)),
                   'upper_boundary': 10 * np.ones((ndim_problem,))}
        options = {'max_function_evaluations': 1e6,
                   'fitness_threshold': 1e-10,
                   'seed_rng': 0,
                   'theta': 0.3,
                   'c': 20,
                   'n_individuals': 200,
                   'verbose_frequency': 20,
                   'record_fitness': True,
                   'record_fitness_frequency': 1}
        if f == schwefel221:
            problem['lower_boundary'] = -100 * np.ones((ndim_problem,))
            problem['upper_boundary'] = 100 * np.ones((ndim_problem,))
        solver = Solver(problem, options)
        results = solver.optimize()
        write_pickle(f.__name__, ndim_problem, results)
        plot(f.__name__)
