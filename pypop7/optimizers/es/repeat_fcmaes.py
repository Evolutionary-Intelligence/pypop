"""Repeat Fig. 4 Fast CMA line from the following paper:
    Z. Li, Q. Zhang, X. Lin, H. zhen
    Fast Covariance Matrix Adaptation for Large-Scale Black-Box Optimization
    IEEE Transaction on Cybernetics vol.50 No.5 May 2020
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8533604
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from pypop7.optimizers.es.fcmaes import FCMAES as Solver
from benchmarks.base_functions import cigar, rosenbrock, discus, ellipsoid, different_powers, _squeeze_and_check

sns.set_theme(style='darkgrid')


def ctb(x):
    x = np.power(_squeeze_and_check(x, True), 2)
    y = x[0] + 1e6 * x[-1]
    for i in range(1, len(x)-1):
        y += 1e4 * x[i]
    return y


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
    result1 = read_pickle(function, 512)
    result2 = read_pickle(function, 1024)
    plt.yscale('log')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    if function == 'cigar' or function == 'ctb':
        plt.ylim([1e-10, 1e10])
        plt.yticks([1e-10, 1e-5, 1e0, 1e5, 1e10])
        plt.xlim([0, 5e5])
        plt.xticks([0, 1e5, 2e5, 3e5, 4e5, 5e5])
    elif function == 'ellipsoid':
        plt.ylim([1e-10, 1e10])
        plt.yticks([1e-10, 1e-5, 1e0, 1e5, 1e10])
        plt.xlim([0, 2.5e7])
        plt.xticks([0, 0.5e7, 1e7, 1.5e7, 2e7, 2.5e7])
    elif function == 'rosenbrock':
        plt.ylim([1e-10, 1e5])
        plt.yticks([1e-10, 1e-5, 1e0, 1e5])
        plt.xlim([0, 5e7])
        plt.xticks([0, 1e7, 2e7, 3e7, 4e7, 5e7])
    elif function == 'discus':
        plt.ylim([1e-10, 1e5])
        plt.yticks([1e-10, 1e-5, 1e0, 1e5])
        plt.xlim([0, 1e7])
        plt.xticks([0, 2e6, 4e6, 6e6, 8e6, 1e7])
    elif function == 'different_powers':
        plt.ylim([1e-10, 1e5])
        plt.yticks([1e-10, 1e-5, 1e0, 1e5])
        plt.xlim([0, 6e6])
        plt.xticks([0, 1e6, 2e6, 3e6, 4e6, 5e6])
    mark_on1 = [int(len(result1['fitness']) / 3), int(len(result1['fitness']) / 3 * 2), len(result1['fitness'])-1]
    mark_on2 = [int(len(result2['fitness']) / 3), int(len(result2['fitness']) / 3 * 2), len(result2['fitness'])-1]
    plt.plot(result1['fitness'][:, 0], result1['fitness'][:, 1], color='r', marker='s',
             markevery=mark_on1, markerfacecolor='none', label='n=512')
    plt.plot(result2['fitness'][:, 0], result2['fitness'][:, 1], color='r', marker='o',
             markevery=mark_on2, markerfacecolor='none', label='n=1024')
    plt.xlabel("evaluations")
    plt.ylabel("objective values")
    plt.title(function.capitalize())
    plt.legend(labels=['n=512', 'n=1024'], loc='best')
    plt.savefig('result_' + function + '.png')
    plt.show()


if __name__ == '__main__':
    # plot Fig. 4
    for f in [cigar, ellipsoid, rosenbrock, discus, ctb, different_powers]:
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        for d in [512, 1024]:
            problem = {'fitness_function': f,
                       'ndim_problem': d,
                       'lower_boundary': -10 * np.ones((d,)),
                       'upper_boundary': 10 * np.ones((d,))}
            options = {'fitness_threshold': 1e-8,
                       'max_function_evaluations': 3e7,
                       'seed_rng': 0,
                       'sigma': 0.1,
                       'verbose_frequency': 2e3,
                       'record_fitness': True,
                       'record_fitness_frequency': 1}
            if f.__name__ == 'cigar':
                options['max_function_evaluations'] = 2e5
            elif f.__name__ == 'ellipsoid':
                options['max_function_evaluations'] = 2e7
            elif f.__name__ == 'rosenbrock':
                options['max_function_evaluations'] = 1.5e7
            elif f.__name__ == 'discus':
                options['max_function_evaluations'] = 4e6
            elif f.__name__ == 'ctb':
                options['max_function_evaluations'] = 2e5
            elif f.__name__ == 'different_powers':
                options['max_function_evaluations'] = 1e6
            solver = Solver(problem, options)
            results = solver.optimize()
            write_pickle(f.__name__, d, results)
        plot(f.__name__)
