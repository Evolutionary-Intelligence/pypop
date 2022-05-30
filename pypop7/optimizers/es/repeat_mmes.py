"""Repeat experiments on four functions [Discus, Cigar, Ellipsoid, Rosenbrock] from the following paper:
    He, X., Zheng, Z. and Zhou, Y., 2021.
    MMES: Mixture model-based evolution strategy for large-scale optimization.
    IEEE Transactions on Evolutionary Computation, 25(2), pp.320-333.
    https://ieeexplore.ieee.org/abstract/document/9244595
"""
import sys
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import discus, cigar, ellipsoid, rosenbrock
from pypop7.optimizers.core import optimizer
from pypop7.optimizers.es.mmes import MMES as Solver

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
    result = read_pickle(function, 1000)
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='r')
    if function == 'discus':
        plt.xlim([1e2, 1e8])
        plt.ylim([1e-8, 1e7])
        plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
        plt.yticks(ticks=[1e-5, 1e0, 1e5])
    elif function == 'cigar':
        plt.xlim([1e2, 1e7])
        plt.ylim([1e-8, 1e10])
        plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6])
        plt.yticks(ticks=[1e-5, 1e0, 1e5])
    elif function == 'ellipsoid':
        plt.xlim([1e2, 1e8])
        plt.ylim([1e-8, 1e9])
        plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
        plt.yticks(ticks=[1e-5, 1e0, 1e5])
    elif function == 'rosenbrock':
        plt.xlim([1e2, 1e8])
        plt.ylim([1e-8, 1e8])
        plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
        plt.yticks(ticks=[1e-5, 1e0, 1e5])
    plt.xlabel("FEs")
    plt.ylabel("Objective Value")
    plt.title(function.capitalize())
    plt.show()


if __name__ == '__main__':
    ndim_problem = 1000
    for f in [discus, cigar, ellipsoid, rosenbrock]:
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -5 * np.ones((ndim_problem,)),
                   'upper_boundary': 5 * np.ones((ndim_problem,))}
        options = {'fitness_threshold': 1e-8,
                   'max_function_evaluations': 1e8,
                   'seed_rng': 2022,  # not given in the original paper
                   'sigma': 3,
                   'verbose_frequency': 20000,
                   'record_fitness': True,
                   'record_fitness_frequency': 1}
        solver = Solver(problem, options)
        results = solver.optimize()
        write_pickle(f.__name__, ndim_problem, results)
        plot(f.__name__)
