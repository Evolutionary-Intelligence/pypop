"""Compare computational efficiency between *DEAP (EDA)* and *PYPOP7 (UMDA)*
    on the well-known Sphere test function.
"""
import os
import sys
import pickle  # for data storage

import numpy as np
import matplotlib.pyplot as plt

from pypop7.optimizers.core import optimizer
sys.modules['optimizer'] = optimizer  # for `pickle`


def read_pickle(s, ii):
    with open(os.path.join('./', s + '-EDA_' + ii + '.pickle'), 'rb') as handle:
        return pickle.load(handle)


if __name__ == '__main__':
    n_trials = 10
    algos = ['PYPOP7', 'DEAP']
    labels = ['PyPop7 (UMDA)', 'DEAP (EDA)']
    colors = ['springgreen', 'blueviolet']
    max_runtime, fitness_threshold = 3600*3 - 10*60, 1e-10
    time, fitness, fe = [], [], []
    for j in range(len(algos)):  # initialize
        time.append([])
        fitness.append([])
        fe.append([])
        for i in range(n_trials):
            time[j].append([])
            fitness[j].append([])
            fe[j].append([])
    for i in range(n_trials):
        for j, a in enumerate(algos):
            results = read_pickle(a, str(i + 1))
            time[j][i] = results['fitness'][:, 0]*results['runtime']/results['n_function_evaluations']
            fe[j][i] = results['fitness'][:, 0]
            y = results['fitness'][:, 1]
            fitness[j][i] = y
            print(i + 1, ' * ', a, ':', results['n_function_evaluations'], results['best_so_far_y'])
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
    top_fitness = [t for t in [tr[2] for tr in top_fitness]]
    print('  #top fitness:', top_fitness)
    print('  #top order:', [(a, to + 1) for a, to in zip(algos, top_order)])
    plt.figure(figsize=(12, 12))
    plt.rcParams['font.family'] = 'Times New Roman'
    for j, a in enumerate(algos):
        plt.plot(fe[j][top_order[j]], fitness[j][top_order[j]],
                 linewidth=5, label=a, color=colors[j])
    plt.xlabel('Number of Function Evaluations', fontsize=30, fontweight='bold')
    plt.ylabel('Cost', fontsize=30, fontweight='bold')
    plt.xticks([0.0e8, 0.2e8, 0.4e8, 0.6e8, 0.8e8, 1.0e8, 1.2e8, 1.4e8, 1.6e8],
               ['0.0e8', '0.2e8', '0.4e8', '0.6e8', '0.8e8', '1.0e8', '1.2e8', '1.4e8', '1.6e8'],
               fontsize=30, fontweight='bold')
    plt.yticks([1e3, 1e4], ['1e3', '1e4'], fontsize=30, fontweight='bold')
    plt.yscale('log')
    plt.title('Sphere', fontsize=30, fontweight='bold')
    plt.legend(labels, fontsize=30)
    plt.savefig('compare_deap-eda_vs_pypop7-umda[cost].eps')
    plt.show()

    algos = ['DEAP', 'PYPOP7']
    labels = ['DEAP (EDA)', 'PyPop7 (UMDA)']
    colors = ['blueviolet', 'springgreen']
    max_runtime, fitness_threshold = 3600 * 3 - 10 * 60, 1e-10
    time, fitness, fe = [], [], []
    for j in range(len(algos)):  # initialize
        time.append([])
        fitness.append([])
        fe.append([])
        for i in range(n_trials):
            time[j].append([])
            fitness[j].append([])
            fe[j].append([])
    for i in range(n_trials):
        for j, a in enumerate(algos):
            results = read_pickle(a, str(i + 1))
            time[j][i] = results['fitness'][:, 0] * results['runtime'] / results['n_function_evaluations']
            fe[j][i] = results['fitness'][:, 0]
            y = results['fitness'][:, 1]
            fitness[j][i] = y
            print(i + 1, ' * ', a, ':', results['n_function_evaluations'], results['best_so_far_y'])
    top_fitness, top_order = [], []
    for j, a in enumerate(algos):
        run, fit, r_f = [], [], []
        for i in range(len(time[j])):
            run.append(time[j][i][-1] if time[j][i][-1] <= max_runtime else max_runtime)
            fit.append(fitness[j][i][-1] if fitness[j][i][-1] >= fitness_threshold else fitness_threshold)
            r_f.append([run[i], fit[i], i])
        r_f.sort(key=lambda x: (x[0], x[1]))  # sort by first runtime then fitness
        order = r_f[int(n_trials / 2)][2]  # for median (but non-standard for simplicity)
        top_order.append(order)
        top_fitness.append([run[order], fit[order], a])
    top_fitness.sort(key=lambda x: (x[0], x[1]))
    top_fitness = [t for t in [tr[2] for tr in top_fitness]]
    fig = plt.figure(figsize=(12, 12))
    plt.rcParams['font.family'] = 'Times New Roman'
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    xticks = []
    for j, a in enumerate(algos):
        ax1.bar([0.5 + j], [fe[j][top_order[j]][-1]], fc=colors[j])
        xticks.append(0.5 + j)
    # ax1.set_xlabel('Open-Source Library', fontsize=30, fontweight='bold')
    ax1.set_ylabel('Number of Function Evaluations', fontsize=30, fontweight='bold')
    ax1.set_xticks(xticks, labels, fontsize=30, fontweight='bold')
    ax1.set_yticks([0.0e8, 0.2e8, 0.4e8, 0.6e8, 0.8e8, 1.0e8, 1.2e8, 1.4e8, 1.6e8],
                   ['0.0e8', '0.2e8', '0.4e8', '0.6e8', '0.8e8', '1.0e8', '1.2e8', '1.4e8', '1.6e8'],
                   fontsize=30, fontweight='bold')
    ax1.set_xlim(0, len(xticks))
    ax2.plot(np.ones(len(xticks) + 1,)*fe[1][top_order[1]][-1]/fe[0][top_order[0]][-1], color='r', linewidth=3)
    ax2.tick_params(colors='r')
    ax2.set_ylabel('Speedup (Function Evaluations)', fontsize=30, fontweight='bold', color='r')
    ax2.set_yticks(np.arange(0, 5, 1), np.arange(0, 5, 1), fontsize=30, fontweight='bold')
    plt.title('Sphere', fontsize=30, fontweight='bold')
    plt.savefig('compare_deap-eda_vs_pypop7-umda[fe].eps')
    plt.show()
