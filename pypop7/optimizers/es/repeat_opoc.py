"""Repeat Figure 2 (Ellipsoid), 4 (Discus) and 5 (Ackley) from the following paper:
    Igel, C., Suttorp, T. and Hansen, N., 2006, July.
    A computational efficient covariance matrix update and a (1+1)-CMA for evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 453-460). ACM.
    https://dl.acm.org/doi/abs/10.1145/1143997.1144082
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pypop7.optimizers.es.opoc import OPOC as Solver
from pypop7.benchmarks.base_functions import discus, ellipsoid, ackley

sns.set_theme(style='darkgrid')


def read_pickle(function, ndim):
    with open(function + '_' + str(ndim) + '.pickle', 'rb') as handle:
        return pickle.load(handle)


def write_pickle(function, ndim, result):
    pickle.dump(result, open(function + '_' + str(ndim) + '.pickle', 'wb'))


def plot(function, ndim):
    plt.figure()
    result = read_pickle(function, ndim)
    if function == 'ellipsoid':
        plt.yscale('log')
        plt.ylim([1e-10, 1e10])
        plt.yticks([1e-10, 1e-5, 1e0, 1e5, 1e10])
        if ndim == 5:
            plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
            plt.xlim([0, 3500])
        elif ndim == 20:
            plt.xticks([0, 2e4, 4e4, 6e4, 8e4])
            plt.xlim([0, 8e4])
    elif function == 'discus':
        plt.yscale('log')
        plt.ylim([1e-25, 1e10])
        plt.yticks([1e-25, 1e-20, 1e-15, 1e-10, 1e-5, 1e0, 1e5, 1e10])
        if ndim == 5:
            plt.xticks([0, 500, 1000, 1500, 2000, 2500])
            plt.xlim([0, 2500])
        elif ndim == 20:
            plt.xticks([0, 0.5e4, 1e4, 1.5e4, 2e4])
            plt.xlim([0, 2.5e4])
    elif function == 'ackley':
        if ndim == 5:
            plt.xticks([0, 100, 200, 300, 400])
            plt.xlim([0, 400])
            plt.yticks([0, 5, 10, 15, 20])
            plt.ylim([0, 23])
        elif ndim == 20:
            plt.xticks([0, 200, 400, 600, 800, 1000])
            plt.xlim([0, 1000])
            plt.yticks([19.8, 20, 20.2, 20.4, 20.6, 20.8, 21, 21.2, 21.4])
            plt.ylim([19.8, 21.4])
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='black')
    plt.xlabel("function evaluations")
    plt.ylabel("function value")
    plt.title(str(ndim) + '-d ' + function.capitalize())
    plt.show()


if __name__ == '__main__':
    seed_rng = np.random.default_rng(0)  # undefined in the original paper
    for f in [ellipsoid, discus, ackley]:
        for d in [5, 20]:
            print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
            problem = {'fitness_function': f,
                       'ndim_problem': d}
            options = {'seed_rng': 0,  # undefined in the original paper
                       'x': seed_rng.uniform(-1, 5, size=(d,)),  # mean
                       'sigma': 3.0,
                       'verbose_frequency': 200,
                       'record_fitness': True,
                       'record_fitness_frequency': 1,
                       'is_restart': False}
            if f.__name__ == 'ellipsoid':
                options['fitness_threshold'] = 1e-10
                if d == 5:
                    options['max_function_evaluations'] = 2500
                elif d == 20:
                    options['max_function_evaluations'] = 6e4
            elif f.__name__ == 'discus':
                options['fitness_threshold'] = 1e-25
                if d == 5:
                    options['max_function_evaluations'] = 2500
                elif d == 20:
                    options['max_function_evaluations'] = 2.5e4
            elif f.__name__ == 'ackley':
                seed_rng = np.random.default_rng(5)  # undefined in the original paper
                options['x'] = seed_rng.uniform(-32.768, 32.768, size=(d,))
                options['sigma'] = 30
                if d == 5:
                    options['max_function_evaluations'] = 400
                elif d == 20:
                    options['max_function_evaluations'] = 1000
            solver = Solver(problem, options)
            results = solver.optimize()
            write_pickle(f.__name__, d, results)
            plot(f.__name__, d)
