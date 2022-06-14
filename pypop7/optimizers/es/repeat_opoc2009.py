"""Repeat Fig.4, Fig.5 in this paper
    (1+1)-Cholesky-CMA-ES (OPOC).
    Reference
    ---------
    Suttorp, T., Hansen, N. and Igel, C., 2009.
    Efficient covariance matrix update for variable metric evolution strategies.
    Machine Learning, 75(2), pp.167-197.
    https://link.springer.com/article/10.1007/s10994-009-5102-1
    (See Algorithm 2 for details.)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from pypop7.optimizers.es.opoc2009 import OPOC2009 as Solver
from pypop7.benchmarks.base_functions import ellipsoid, cigar

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


# plot fig4
def plot(function, problem_dim):
    plt.figure()
    result = read_pickle(function, problem_dim)
    plt.yscale('log')
    plt.ylim([1e-250, 1e50])
    plt.yticks([1e-250, 1e-200, 1e-150, 1e-100, 1e-50, 1e0, 1e50])
    plt.xticks([0, 2e4, 4e4, 6e4, 8e4, 1e5])
    plt.xlim([0, 1e5])
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='green')
    plt.xlabel("objective function evaluations")
    plt.ylabel("objective function value")
    plt.title(function.capitalize())
    plt.show()


if __name__ == '__main__':
    # fig.4
    for f in [ellipsoid]:
        for d in [20]:
            print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
            problem = {'fitness_function': f,
                       'ndim_problem': d,
                       'lower_boundary': -5 * np.ones((d,)),
                       'upper_boundary': 5 * np.ones((d,))}
            options = {'fitness_threshold': 1e-250,
                       'max_runtime': 3600,  # 1 hours
                       'seed_rng': 0,
                       'x': 4 * np.ones((d,)),  # mean
                       'sigma': 3.0,
                       'max_function_evaluations': 6e4,
                       'verbose_frequency': 2000,
                       'record_fitness': True,
                       'record_fitness_frequency': 1,
                       'is_restart': False}
            solver = Solver(problem, options)
            results = solver.optimize()
            write_pickle(f.__name__, d, results)
            plot(f.__name__, d)
    # fig 5
    evaluations = []
    ds = [5, 10, 15, 20, 40, 80, 160, 320]
    for f in [cigar]:
        for d in ds:
            problem = {'fitness_function': f,
                       'ndim_problem': d,
                       'lower_boundary': -5 * np.ones((d,)),
                       'upper_boundary': 5 * np.ones((d,))}
            options = {'fitness_threshold': 1e-15,
                       'max_runtime': 3600,  # 1 hours
                       'seed_rng': 0,
                       'x': 4 * np.ones((d,)),  # mean
                       'sigma': 3.0,
                       'max_function_evaluations': 2e5,
                       'verbose_frequency': 2000,
                       'record_fitness': False,
                       'record_fitness_frequency': 1,
                       'is_restart': False}
            solver = Solver(problem, options)
            results = solver.optimize()
            evaluations.append(results['n_function_evaluations'])
    plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([1e3, 1e7])
    plt.yticks([1e3, 1e4, 1e5, 1e6, 1e7])
    plt.xlim([1, 4e2])
    plt.xticks([10, 100])
    plt.plot(ds, evaluations, 'ro--', color='green')
    plt.xlabel('search space dimension n')
    plt.ylabel('objective function evaluations')
    plt.show()
