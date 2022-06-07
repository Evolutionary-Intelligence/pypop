"""Repeat Fig. 3 (Ellipsoid, Discus, Rosenbrock) from the following paper:
    Li, Z. and Zhang, Q., 2018.
    A simple yet efficient evolution strategy for large-scale black-box optimization.
    IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
    https://ieeexplore.ieee.org/abstract/document/8080257
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pypop7.benchmarks.base_functions import ellipsoid, discus, rosenbrock
from pypop7.optimizers.es.rmes import RMES

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


def plot(function, ndim):
    plt.figure()
    result = read_pickle(function, ndim)
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='orange')
    plt.xlim([1e2, 1e8])
    plt.xticks([1e2, 1e4, 1e6, 1e8])
    if function == 'ellipsoid':
        plt.ylim([1e-8, 1e10])
        plt.yticks(ticks=[1e-8, 1e-4, 1e0, 1e4, 1e8])
    elif function == 'discus':
        plt.ylim([1e-8, 1e6])
        plt.yticks(ticks=[1e-8, 1e-4, 1e0, 1e4])
    elif function == 'rosenbrock':
        plt.ylim([1e-8, 1e10])
        plt.yticks(ticks=[1e-8, 1e-4, 1e0, 1e4, 1e8])
    plt.xlabel("Function Evaluations")
    plt.ylabel("Objective Value")
    plt.title(function.capitalize())
    plt.show()


if __name__ == '__main__':
    ndim_problem = 1000
    for f in [ellipsoid, discus, rosenbrock]:
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -10 * np.ones((ndim_problem,)),
                   'upper_boundary': 10 * np.ones((ndim_problem,))}
        options = {'fitness_threshold': 1e-8,
                   'max_function_evaluations': 1e8,
                   'seed_rng': 2022,  # not given in the original paper
                   'sigma': 20 / 3,
                   'verbose_frequency': 20000,
                   'record_fitness': True,
                   'record_fitness_frequency': 1}
        rmes = RMES(problem, options)
        results = rmes.optimize()
        write_pickle(f.__name__, ndim_problem, results)
        plot(f.__name__, ndim_problem)
