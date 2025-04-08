"""Compare computational efficiency between *DEAP (SES)* and
    *PYPOP7 (SES)* on the well-known Sphere function.

    Chinese: 该绘图脚本被段琦琦的博士论文（哈工大与南科大联合培养）所使用。
"""
import os
import sys
import pickle5 as pickle

import numpy as np
import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt

from pypop7.optimizers.core import optimizer
sys.modules['optimizer'] = optimizer  # for `pickle`


def read_pickle(s, ii):
    file_name = os.path.join('./', s + '-SES_' + ii + '.pickle')
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)


if __name__ == '__main__':
    sns.set_theme(style='darkgrid', rc={'figure.figsize':(2.4, 2.4)})
    plt.rcParams['font.size'] = '10'
    plt.rcParams['font.family'] = 'Times New Roman'
    # fontsize=30, fontweight='bold'

    n_trials = 10
    algos = ['PYPOP7', 'DEAP']
    labels = ['PyPop7-ES', 'DEAP-ES']
    colors = ["#F08C55", "#6EC8C8"]
    max_runtime, fitness_threshold = 3600 * 3 - 10 * 60, 1e-10
    time, fitness, fe = [], [], []
    for i in range(len(algos)):  # initialize
        time.append([])
        fitness.append([])
        fe.append([])
        for j in range(n_trials):
            time[i].append([])
            fitness[i].append([])
            fe[i].append([])
    for i in range(n_trials):
        for j, a in enumerate(algos):
            results = read_pickle(a, str(i + 1))
            time[j][i] = results['fitness'][:, 0] *\
                results['runtime'] /\
                    results['n_function_evaluations']
            fe[j][i] = results['fitness'][:, 0]
            fitness[j][i] = results['fitness'][:, 1]
    top_fitness, top_order = [], []
    for j, a in enumerate(algos):
        run, fit, r_f = [], [], []
        for i in range(len(time[j])):
            run.append(time[j][i][-1] if time[j][i][-1] <= max_runtime else max_runtime)
            fit.append(fitness[j][i][-1] if fitness[j][i][-1] >= fitness_threshold else fitness_threshold)
            r_f.append([run[i], fit[i], i])
        r_f.sort(key=lambda x: (x[0], x[1]))  # sort by first runtime then fitness
        order = r_f[int(n_trials/2)][2]  # for median (but non-standard for simplicity)
        top_order.append(order)
        top_fitness.append([run[order], fit[order], a])
    top_fitness.sort(key=lambda x: (x[0], x[1]))
    top_fitness = [t for t in [tf[2] for tf in top_fitness]]
    print('  #top fitness:', top_fitness)
    print('  #top order:', [(a, to + 1) for a, to in zip(algos, top_order)])
    for j, a in enumerate(algos):
        plt.plot(fe[j][top_order[j]], fitness[j][top_order[j]],
                 linewidth=3, label=a, color=colors[j])
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Fitness (to be Minimized)')
    plt.xticks([0.0e8, 0.4e8, 0.8e8, 1.2e8, 1.6e8],
               ['0.0e8', '0.4e8', '0.8e8', '1.2e8', '1.6e8'])
    plt.title('Sphere')
    plt.legend(labels)
    plt.yscale('log')
    plt.savefig('compare_deap-ses_vs_pypop7-ses[cost].png',
                dpi=700, bbox_inches='tight')
    plt.show()

    sns.set_theme(style='dark')
    plt.rcParams['font.size'] = '10'
    plt.rcParams['font.family'] = 'Times New Roman'
    fig = plt.figure(figsize=(2.4, 2.4))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    xticks = []
    for j, a in enumerate(algos):
        ax1.bar([0.5 + j], [fe[j][top_order[j]][-1]], fc=colors[j])
        xticks.append(0.5 + j)
    ax1.set_ylabel('Number of Evaluations')
    ax1.set_xticks(xticks, labels)
    ax1.set_yticks([0.0e8, 0.4e8, 0.8e8, 1.2e8, 1.6e8],
                   ['0.0e8', '0.4e8', '0.8e8', '1.2e8', '1.6e8'])
    ax1.set_xlim(0, len(xticks))
    ax2.plot(np.ones(len(xticks) + 1,) *\
             fe[0][top_order[0]][-1] /\
                fe[1][top_order[1]][-1])
    ax2.tick_params(colors='m')
    ax2.set_ylabel('Speedup', color='m', labelpad=-1)
    ax2.set_yticks(np.arange(0, 30, 5), np.arange(0, 30, 5))
    plt.title('Sphere')
    plt.savefig('compare_deap-ses_vs_pypop7-ses[fe].png',
                dpi=700, bbox_inches='tight')
    plt.show()
