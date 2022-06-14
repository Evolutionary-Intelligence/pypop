"""Repeat Figure 2, 4 in paper
    (1+1)-Cholesky-CMA-ES (OPOC).
    Reference
    ---------
    Igel, C., Suttorp, T. and Hansen, N., 2006, July.
    A computational efficient covariance matrix update and a (1+1)-CMA for evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 453-460). ACM.
    https://dl.acm.org/doi/abs/10.1145/1143997.1144082
    (See Algorithm 2 for details.)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from pypop7.optimizers.es.opoc import OPOC as Solver
from pypop7.benchmarks.base_functions import discus, ellipsoid, ackley

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
    if function == 'ellipsoid':
        plt.yscale('log')
        plt.ylim([1e-10, 1e10])
        plt.yticks([1e-10, 1e-5, 1e0, 1e5, 1e10])
        if problem_dim == 5:
            plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
            plt.xlim([0, 3500])
        elif problem_dim == 20:
            plt.xticks([0, 2e4, 4e4, 6e4, 8e4])
            plt.xlim([0, 8e4])
    elif function == 'discus':
        plt.yscale('log')
        plt.ylim([1e-25, 1e10])
        plt.yticks([1e-25, 1e-20, 1e-15, 1e-10, 1e-5, 1e0, 1e5, 1e10])
        if problem_dim == 5:
            plt.xticks([0, 500, 1000, 1500, 2000, 2500])
            plt.xlim([0, 2500])
        elif problem_dim == 20:
            plt.xticks([0, 0.5e4, 1e4, 1.5e4, 2e4])
            plt.xlim([0, 2.5e4])
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='black')
    plt.xlabel("function evaluations")
    plt.ylabel("function value")
    plt.title(function.capitalize() + " Dimension: " + str(problem_dim))
    plt.show()


if __name__ == '__main__':
    for f in [ellipsoid, discus, ackley]:
        for d in [5, 20]:
            print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
            problem = {'fitness_function': f,
                       'ndim_problem': d,
                       'lower_boundary': -5 * np.ones((d,)),
                       'upper_boundary': 5 * np.ones((d,))}
            options = {'fitness_threshold': 1e-10,
                       'max_runtime': 3600,  # 1 hours
                       'seed_rng': 0,
                       'x': 4 * np.ones((d,)),  # mean
                       'sigma': 3.0,
                       'verbose_frequency': 200,
                       'record_fitness': True,
                       'record_fitness_frequency': 1,
                       'is_restart': False}
            if f.__name__ == 'discus':
                options['fitness_threshold'] = 1e-25
                if d == 5:
                    options['max_function_evaluations'] = 2500
                elif d == 20:
                    options['max_function_evaluations'] = 2.5e4
            elif f.__name__ == 'ellipsoid':
                if d == 5:
                    options['max_function_evaluations'] = 2500
                elif d == 20:
                    options['max_function_evaluations'] = 6e4
            solver = Solver(problem, options)
            results = solver.optimize()
            write_pickle(f.__name__, d, results)
            plot(f.__name__, d)
