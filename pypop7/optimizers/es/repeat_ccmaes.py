"""Repeat Fig.6, Fig.7 in paper
    Cholesky-CMA-ES (CCMAES, (μ/μ_w,λ)-Cholesky-CMA-ES).
    Reference
    ---------
    Suttorp, T., Hansen, N. and Igel, C., 2009.
    Efficient covariance matrix update for variable metric evolution strategies.
    Machine Learning, 75(2), pp.167-197.
    https://link.springer.com/article/10.1007/s10994-009-5102-1
    (See Algorithm 4 for details.)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from pypop7.optimizers.es.ccmaes import CCMAES as Solver
from pypop7.benchmarks.base_functions import _squeeze_and_check, ellipsoid, rosenbrock

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
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='black')
    plt.xlabel("objective function evaluations")
    plt.ylabel("objective function value")
    plt.title(function.capitalize())
    plt.show()


if __name__ == '__main__':
    for f in [ellipsoid, rosenbrock]:
        for d in [20]:
            print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
            problem = {'fitness_function': f,
                       'ndim_problem': d,
                       'lower_boundary': -5 * np.ones((d,)),
                       'upper_boundary': 5 * np.ones((d,))}
            options = {'fitness_threshold': 1e-10,
                       'max_runtime': 3600,  # 1 hours
                       'seed_rng': 0,
                       'x': 0.2 * np.ones((d,)),  # mean
                       'sigma': 0.2/3,
                       'verbose_frequency': 2000,
                       'record_fitness': True,
                       'record_fitness_frequency': 1,
                       'is_restart': False}
            if f.__name__ == 'ellipsoid':
                options['fitness_threshold'] = 1e-30
                options['max_function_evaluations'] = 3e4
            elif f.__name__ == 'rosenbrock':
                options['fitness_threshold'] = 1e-15
                options['max_function_evaluations'] = 3e4
            solver = Solver(problem, options)
            results = solver.optimize()
            write_pickle(f.__name__, d, results)
            plot(f.__name__, d)
