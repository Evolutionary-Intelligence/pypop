"""Repeat Fig.3(left) from the following paper:
     Loshchilov, I., 2014, July.
    A computationally efficient limited memory CMA-ES for large scale optimization.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 397-404). ACM.
    https://dl.acm.org/doi/abs/10.1145/2576768.2598294
"""
import sys
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.rotated_functions import ellipsoid, generate_rotation_matrix
from pypop7.optimizers.core import optimizer
from pypop7.optimizers.es.lmcmaes import LMCMAES as Solver

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


def plot(function, problem_dim):
    plt.figure()
    result = read_pickle(function, problem_dim)
    print(result)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([1e-10, 1e10])
    plt.yticks([1e-10, 1e-5, 1e0, 1e5, 1e10])
    plt.xlim([1e1, 1e7])
    plt.xticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='purple')
    plt.xlabel("Function Evaluations")
    plt.ylabel("Objective Function")
    plt.title(function.capitalize())
    plt.show()


if __name__ == '__main__':
    for f in [ellipsoid]:
        for d in [128]:
            problem = {'fitness_function': f,
                       'ndim_problem': d,
                       'lower_boundary': -5 * np.ones((d,)),
                       'upper_boundary': 5 * np.ones((d,))}
            options = {'max_function_evaluations': 1e9,
                       'fitness_threshold': 1e-10,
                       'seed_rng': 0,  # undefined in the original paper
                       'x': 4 * np.ones((d,)),  # mean
                       'sigma': 5,
                       'verbose_frequency': 5000,
                       'record_fitness': True,
                       'record_fitness_frequency': 1,
                       'is_restart': False}
            solver = Solver(problem, options)
            results = solver.optimize()
            write_pickle(f.__name__, d, results)
            plot(f.__name__, d)
