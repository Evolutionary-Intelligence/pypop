"""This Python script plots the *median* convergence curves for various EDAs
    with respective to actual runtime (to be needed).

    https://pypop.readthedocs.io/en/latest/index.html
    https://deap.readthedocs.io/en/master/
"""
import os
import sys
import pickle  # for data storage

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

from pypop7.optimizers.core import optimizer
sys.modules['optimizer'] = optimizer  # for `pickle`


def read_pickle(s, ff, ii):
    afile = os.path.join('./docs/pypop7_benchmarks_lso/single-core', s + '/Algo-' + s + '_Func-' + ff + '_Dim-2000_Exp-' + ii + '.pickle')
    with open(afile, 'rb') as handle:
        return pickle.load(handle)


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = '12'

    n_trials = 4  # number of trials (independent experiments)
    algos = ['DEAPEDA', 'EMNA', 'AEMNA', 'UMDA', 'RPEDA', 'DCEM', 'DSCEM', 'MRAS', 'SCEM']
    max_runtime, fitness_threshold = 3600*3 - 10*60, 1e-10
    colors, c = [name for name, _ in colors.cnames.items()], []
    for i in [7] + list(range(9, 18)) + list(range(19, len(colors))):
        c.append(colors[i])  # for better colors
    funcs = ['cigar', 'cigar_discus', 'ackley', 'bohachevsky',
             'different_powers', 'discus', 'griewank', 'levy_montalvo',
             'ellipsoid', 'rosenbrock', 'michalewicz', 'rastrigin',
             'schwefel12', 'schwefel221', 'salomon', 'scaled_rastrigin',
             'step', 'schaffer', 'sphere', 'skew_rastrigin']

    # fig
    fig = plt.figure(figsize=(24, 22))  # (16, 9)
    no_of_rows, no_of_cols = 5, 4
    axs = fig.subplots(no_of_rows, no_of_cols)
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
                if f == "michalewicz":
                    y += 600
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

        # plot
        m, n = divmod(k, no_of_cols)
        axs[m][n].set_yscale('log')
        for j, a in enumerate(algos):
            axs[m][n].plot(time[j][top_order[j]], fitness[j][top_order[j]], label=a, color=c[j])
        axs[m][n].set_title(f, fontsize=17, fontweight="bold")
        x_label = axs[m][n].get_xticklabels()
        [x_label_temp.set_fontsize(14) for x_label_temp in x_label]
        [x_label_temp.set_fontweight("bold") for x_label_temp in x_label]
        y_label = axs[m][n].get_yticklabels()
        [y_label_temp.set_fontsize(14) for y_label_temp in y_label]
        [y_label_temp.set_fontweight("bold") for y_label_temp in y_label]

    lines, labels = axs[-1][-1].get_legend_handles_labels()
    leg = fig.legend(lines, labels, loc='center', ncol=7, fontsize=25, bbox_to_anchor=(0.5, 0.92))  # ncol=9
    for text in leg.get_texts():
        text.set_fontweight('bold')
    fig.text(0.08, 0.5, 'Fitness (Minimized)', va='center', rotation='vertical', fontsize='xx-large', fontweight="bold")
    fig.text(0.5, 0.08, 'Running Time (Seconds)', va='center', ha='center', fontsize='xx-large', fontweight="bold")
    plt.savefig('EDAs.png', dpi=300, bbox_inches='tight')
    plt.show()
