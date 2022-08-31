"""Repeat the following paper:
    Loshchilov, I., 2014, July.
    A computationally efficient limited memory CMA-ES for large scale optimization.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 397-404). ACM.
    https://dl.acm.org/doi/abs/10.1145/2576768.2598294
"""
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import ellipsoid
from pypop7.benchmarks.rotated_functions import generate_rotation_matrix
from pypop7.benchmarks.rotated_functions import ellipsoid as rotated_ellipsoid
from pypop7.optimizers.es.lmcmaes import LMCMAES as Solver

sns.set_theme(style='darkgrid')


def plot(function, problem_dim):
    plt.figure()
    result = pickle.load(open(function + '_' + str(problem_dim) + '.pickle', 'rb'))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-10, 1e10])
    plt.yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8, 1e10])
    plt.xlim([1e2, 2e8])
    plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='blue')
    plt.xlabel("Function Evaluations")
    plt.ylabel("Objective Function")
    plt.title(function.capitalize())
    plt.show()


if __name__ == '__main__':
    for f in [ellipsoid, rotated_ellipsoid]:
        for dim in [2*128]:
            problem = {'fitness_function': f,
                       'ndim_problem': dim,
                       'lower_boundary': -5.0*np.ones((dim,)),
                       'upper_boundary': 5.0*np.ones((dim,))}
            options = {'fitness_threshold': 1e-10,
                       'seed_rng': 1,  # undefined in the original paper
                       'sigma': 5.0,
                       'verbose_frequency': 2e4,
                       'record_fitness': True,
                       'record_fitness_frequency': 1,
                       'is_restart': False}
            if f in [rotated_ellipsoid]:
                generate_rotation_matrix(f, dim, 0)
                function_name = 'rotated_' + f.__name__
            else:
                function_name = f.__name__
            solver = Solver(problem, options)
            results = solver.optimize()
            print(results['fitness'])
            pickle.dump(results, open(function_name + '_' + str(dim) + '.pickle', 'wb'))
            plot(function_name, dim)
