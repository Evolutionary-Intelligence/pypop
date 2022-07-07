"""Repeat Fig.3(Right) from the following paper:
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


def read_pickle(function):
    file = function + '.pickle'
    with open(file, 'rb') as handle:
        result = pickle.load(handle)
        return result


def write_pickle(function, result):
    file = open(function + '.pickle', 'wb')
    pickle.dump(result, file)
    file.close()


def plot(function):
    plt.figure()
    result = read_pickle(function)
    result = np.array(result)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([1e4, 1e10])
    plt.yticks([1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10])
    plt.xlim([1e2, 1e4])
    plt.xticks([1e2, 1e3, 1e4])
    plt.plot(result[:, 0], result[:, 1], color='purple', marker='s')
    plt.xlabel("Dimension")
    plt.ylabel("Function Evaluations")
    plt.title(function.capitalize())
    plt.show()


if __name__ == '__main__':
    function_evaluations = []
    for f in [ellipsoid]:
        for d in [128, 256, 512, 1024, 2048, 4096, 8192]:
            problem = {'fitness_function': f,
                       'ndim_problem': d,
                       'lower_boundary': -5 * np.ones((d,)),
                       'upper_boundary': 5 * np.ones((d,))}
            options = {'max_function_evaluations': 1e10,
                       'fitness_threshold': 1e-10,
                       'seed_rng': 0,  # undefined in the original paper
                       'x': 4 * np.ones((d,)),  # mean
                       'sigma': 5,
                       'verbose_frequency': 5000,
                       'record_fitness': False,
                       'record_fitness_frequency': 1,
                       'is_restart': False}
            generate_rotation_matrix(f, d, 2022)
            solver = Solver(problem, options)
            results = solver.optimize()
            function_evaluations.append([d, results['n_function_evaluations']])
        write_pickle(f.__name__, function_evaluations)
        plot(f.__name__)
