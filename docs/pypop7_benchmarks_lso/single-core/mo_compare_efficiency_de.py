"""This Python script plots the *median* convergence curves for various DEs
    with respective to actual runtime (to be needed).

    https://pypop.readthedocs.io/en/latest/index.html
    https://deap.readthedocs.io/en/master/
"""
import os
import sys
import pickle  # for data storage

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

    n_trials = 4  # number of trials (independent experiments)
    algos = ['DEAPDE', 'CDE', 'TDE', 'CODE', 'JADE', 'SHADE']
    max_runtime, fitness_threshold = 3600*3 - 10*60, 1e-10
    funcs = ['griewank', 'ackley', 'rastrigin', 'levy_montalvo', 'michalewicz',
             'salomon', 'bohachevsky', 'scaled_rastrigin', 'skew_rastrigin', 'schaffer']
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
            for j, a in enumerate(algos):
                results = read_pickle(a, f, str(i + 1))
                time[j][i] = results['fitness'][:, 0]*results['runtime']/results['n_function_evaluations']
                y = results['fitness'][:, 1]
                if f == 'michalewicz':  # for printing in the log scale
                    y += 600.0
                fitness[j][i] = y
                print(' '*4, i + 1, ' + ', a, ':', results['best_so_far_y'], results['n_function_evaluations'])
        top_fitness, top_order = [], []
        for j, a in enumerate(algos):
            run, fit, r_f = [], [], []
            for i in range(len(time[j])):
                run.append(time[j][i][-1] if time[j][i][-1] <= max_runtime else max_runtime)
                fit.append(fitness[j][i][-1] if fitness[j][i][-1] >= fitness_threshold else fitness_threshold)
                r_f.append([run[i], fit[i], i])
            r_f.sort(key=lambda x: (x[0], x[1]))  # sort by first runtime then fitness
            order = r_f[int(n_trials/2)][2]  # for median (but non-standard)
            top_order.append(order)
            top_fitness.append([run[order], fit[order], a])
        top_fitness.sort(key=lambda x: (x[0], x[1]))
        top_fitness = [t for t in [tr[2] for tr in top_fitness]]
        print('  #top fitness:', top_fitness)
        print('  #top order:', [(a, to + 1) for a, to in zip(algos, top_order)])
        plt.figure(figsize=(8.5, 8.5))
        plt.yscale('log')
        for j, a in enumerate(algos):
            plt.plot(time[j][top_order[j]], fitness[j][top_order[j]], label=a)
        plt.xlabel('Running Time (Seconds)', fontsize=24, fontweight='bold')
        plt.ylabel('Cost', fontsize=24, fontweight='bold')
        plt.title(f, fontsize=24, fontweight='bold')
        plt.xticks(fontsize=22, fontweight='bold')
        plt.yticks(fontsize=22, fontweight='bold')
        plt.legend(loc='best', ncol=3, fontsize=14)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.savefig('./figures/DEs-' + f + '.eps')
        plt.show()
