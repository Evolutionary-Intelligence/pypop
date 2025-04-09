"""This script plots the *median* convergence curves for
    various DE w.r.t. actual runtime (to be needed).

    Chinese: 该绘图脚本被段琦琦的博士论文（哈工大与南科大联合培养）所使用。
"""
import os
import sys
import pickle5 as pickle

import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.optimizers.core import optimizer
sys.modules['optimizer'] = optimizer


def read_pickle(a, f, i):
    folder = './docs/pypop7_benchmarks_lso/single-core'
    file_name = a + '/Algo-' + a + '_Func-' + f +\
        '_Dim-2000_Exp-' + i + '.pickle'
    with open(os.path.join(folder, file_name), 'rb') as handle:
        return pickle.load(handle)


if __name__ == '__main__':
    font_size = 11
    sns.set_theme(style='darkgrid')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = '11'

    n_trials = 4  # number of trials (independent experiments)
    algos = ['DEAPDE', 'CDE', 'TDE', 'CODE', 'JADE', 'SHADE']
    max_runtime, fitness_threshold = 3600.0 * 3 - 10.0 * 60, 1e-10
    # funcs = ['cigar', 'cigar_discus', 'ackley', 'bohachevsky',
    #          'different_powers', 'discus', 'griewank', 'levy_montalvo',
    #          'ellipsoid', 'rosenbrock', 'michalewicz', 'rastrigin',
    #          'schwefel12', 'schwefel221', 'salomon', 'scaled_rastrigin',
    #          'step', 'schaffer', 'sphere', 'skew_rastrigin']
    funcs = ['cigar', 'ackley', 'bohachevsky',
             'discus', 'ellipsoid', 'rosenbrock',
             'rastrigin', 'schwefel12', 'schwefel221']
    fig = plt.figure(figsize=(9, 11))
    no_of_rows, no_of_cols = 3, 3 # 5, 4
    axs = fig.subplots(no_of_rows, no_of_cols)
    for k, f in enumerate(funcs):
        time, fitness = [], []
        for i in range(len(algos)):
            time.append([])
            fitness.append([])
            for _ in range(n_trials):
                time[i].append([])
                fitness[i].append([])
        for i in range(n_trials):
            for j, a in enumerate(algos):
                results = read_pickle(a, f, str(i + 1))
                time[j][i] = results['fitness'][:, 0] *\
                    results['runtime'] /\
                        results['n_function_evaluations']
                y = results['fitness'][:, 1]
                if f == 'michalewicz':
                    y += 600  # to plot log-scale y-axis
                fitness[j][i] = y
        top_order = []
        for j, a in enumerate(algos):
            run, fit, r_f = [], [], []
            for i in range(len(time[j])):
                run.append(time[j][i][-1] if time[j][i][-1] <= max_runtime else max_runtime)
                fit.append(fitness[j][i][-1] if fitness[j][i][-1] >= fitness_threshold else fitness_threshold)
                r_f.append([run[i], fit[i], i])
            r_f.sort(key=lambda x: (x[0], x[1]))
            order = r_f[int(n_trials / 2)][2]  # for median (but non-standard)
            top_order.append(order)
        m, n = divmod(k, no_of_cols)
        axs[m][n].set_yscale('log')
        for j, a in enumerate(algos):
            axs[m][n].plot(time[j][top_order[j]],
                           fitness[j][top_order[j]],
                           label=a)
        axs[m][n].set_title(f, fontsize=font_size)  # , fontweight='bold'
        x_label = axs[m][n].get_xticklabels()
        [xl.set_fontsize(font_size) for xl in x_label]
        # [xl.set_fontweight('bold') for xl in x_label]
        y_label = axs[m][n].get_yticklabels()
        [yl.set_fontsize(font_size) for yl in y_label]
        # [yl.set_fontweight('bold') for yl in y_label]

    lines, labels = axs[-1][-1].get_legend_handles_labels()
    leg = fig.legend(lines, labels, loc='center', ncol=6,
                     fontsize=font_size,
                     bbox_to_anchor=(0.51, 0.93))
    # for text in leg.get_texts():
    #     text.set_fontweight('bold')
    fig.text(0.05, 0.5, 'Fitness (Minimized)', va='center',
             rotation='vertical', fontsize=font_size)  # 'xx-large'
    fig.text(0.5, 0.05, 'Runtime (Seconds)', va='center',
             ha='center', fontsize=font_size)  # 'xx-large'
    plt.savefig('DE.png', dpi=700, bbox_inches='tight')
    plt.show()
