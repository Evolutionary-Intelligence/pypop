"""This Python script plots the *median* convergence curves for DE
    with respective to (actual) runtime.

    https://pypop.readthedocs.io/en/latest/index.html
    https://deap.readthedocs.io/en/master/
"""
import os
import sys
import pickle  # for data storage

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.optimizers.core import optimizer
sys.modules['optimizer'] = optimizer  # for `pickle`


def read_pickle(s, ff, ii):
    afile = os.path.join('./', s + '/Algo-' + s + '_Func-' + ff + '_Dim-2000_Exp-' + ii + '.pickle')
    with open(afile, 'rb') as handle:
        return pickle.load(handle)


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')

    n_trials = 10  # number of trials (independent experiments)
    algos = ['DEAPDE', 'CDE', 'TDE', 'CODE', 'JADE', 'SHADE']
    max_runtime, fitness_threshold = 3600*3 - 10*60, 1e-10
    funcs = ['sphere', 'cigar', 'discus', 'cigar_discus', 'ellipsoid',
             'different_powers', 'schwefel221', 'step', 'rosenbrock', 'schwefel12']
    for k, f in enumerate(funcs):
        print('* {:s} ***'.format(f))
        time, fitness = [], []
        for j in range(len(algos)):  # initialize
            time.append([])
            fitness.append([])
            for i in range(n_trials):
                time[j].append([])
                fitness[j].append([])
        for i in range(n_trials):
            b = []
            for j, a in enumerate(algos):
                results = read_pickle(a, f, str(i + 1))
                b.append(results['best_so_far_y'])
                time[j][i] = results['fitness'][:, 0]*results['runtime']/results['n_function_evaluations']
                y = results['fitness'][:, 1]
                for i_y in range(1, len(y)):  # for best-so-far fitness curve
                    if y[i_y] > y[i_y - 1]:
                        y[i_y] = y[i_y - 1]
                fitness[j][i] = y
        plt.figure(figsize=(8.6, 8.6))
        plt.yscale('log')
        top_ranked = []
        for j, a in enumerate(algos):
            end_runtime = [time[j][i][-1] for i in range(len(time[j]))]
            end_fit = [fitness[j][i][-1] for i in range(len(fitness[j]))]
            order = np.argsort(end_runtime)[int(n_trials/2)]  # for median (but non-standard)
            _r = end_runtime[order] if end_runtime[order] <= max_runtime else max_runtime
            _f = end_fit[order] if end_fit[order] >= fitness_threshold else fitness_threshold
            top_ranked.append([_r, _f, a])
        top_ranked.sort(key=lambda x: (x[0], x[1]))  # sort by first runtime then fitness
        top_ranked = [t for t in [tr[2] for tr in top_ranked]]
        print('  #top:', top_ranked)
        for j, a in enumerate(algos):
            end_runtime = [time[j][i][-1] for i in range(len(time[j]))]
            order = np.argsort(end_runtime)[int(n_trials/2)]  # for median (but non-standard)
            plt.plot(time[j][order], fitness[j][order], label=a)
        plt.xlabel('Running Time (Seconds)', fontsize=24, fontweight='bold')
        plt.ylabel('Cost', fontsize=24, fontweight='bold')
        plt.title(f, fontsize=24, fontweight='bold')
        plt.xticks(fontsize=22, fontweight='bold')
        plt.yticks(fontsize=22, fontweight='bold')
        plt.legend(loc=4, ncol=3, fontsize=12)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.savefig('./figures/DEs-' + f + '.png', format='png')
        plt.show()
