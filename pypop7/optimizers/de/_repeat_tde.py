"""Repeat the following paper for `TDE`:
    Fan, H.Y. and Lampinen, J., 2003.
    A trigonometric mutation operation to differential evolution.
    Journal of Global Optimization, 27(1), pp.105-129.
    https://link.springer.com/article/10.1023/A:1024653025686
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from pypop7.benchmarks.base_functions import ackley, rastrigin
from pypop7.optimizers.de.tde import TDE as Solver

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
    result['fitness'][:, 0] *= result['runtime'] / result['n_function_evaluations']
    if function == 'ackley':
        plt.ylim([0, 20])
        plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        plt.xlim([0, 45])
        plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    elif function == 'rastrigin':
        plt.ylim([0, 160])
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140, 160])
        plt.xlim([0, 30])
        plt.xticks([0, 5, 10, 15, 20, 25, 30])
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='black')
    plt.xlabel("CPU Time Seconds")
    if function == 'ackley':
        plt.ylabel("f1(x)")
    elif function == 'rastrigin':
        plt.ylabel("f2(x)")
    plt.title(function.capitalize())
    plt.show()


if __name__ == '__main__':
    for f in [ackley, rastrigin]:
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        if f.__name__ == 'ackley':
            ndim_problem = 30
        elif f.__name__ == 'rastrigin':
            ndim_problem = 20
        ndim_problem = 30
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem}
        options = {'max_function_evaluations': 3e5,
                   'fitness_threshold': 1e-10,
                   'seed_rng': 0,
                   'saving_fitness': 1,
                   'verbose': 2000}
        if f.__name__ == 'ackley':
            problem['lower_boundary'] = -20 * np.ones((ndim_problem,))
            problem['upper_boundary'] = 30 * np.ones((ndim_problem,))
        elif f.__name__ == 'rastrigin':
            problem['lower_boundary'] = -5.12 * np.ones((ndim_problem,))
            problem['upper_boundary'] = 5.12 * np.ones((ndim_problem,))
        solver = Solver(problem, options)
        results = solver.optimize()
        write_pickle(f.__name__, ndim_problem, results)
        plot(f.__name__, ndim_problem)
