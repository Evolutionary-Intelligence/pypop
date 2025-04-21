"""This script has been used in Qiqi Duan's Ph.D. Dissertation (HIT&SUSTech).

    Chinese: 该绘图脚本被段琦琦的博士论文（哈工大与南科大联合培养）所使用。
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pypop7.benchmarks.utils import generate_xyz
# abstract class for all evolution Strategies
from pypop7.optimizers.es.es import ES
# Matrix Adaptation Evolution Strategy
from pypop7.optimizers.es.maes import MAES


matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10  # 对应5号字体


def cd(x):  # from https://arxiv.org/pdf/1610.00040v1.pdf
    return 7.0 * (x[0] ** 2) + 6.0 * x[0] * x[1] + 8.0 * (x[1] ** 2)


# helper function for 2D-plotting
def plot_contour(func, x, y):
    x, y, z = generate_xyz(func, x, y, 200)
    plt.contourf(x, y, z, cmap='bone')
    plt.contour(x, y, z, colors='white')


def plot(xs, means, bound=[-10.0, 10.0]):
    for i in range(len(xs)):
        plt.figure(figsize=(2.5, 2.5))
        plt.title('不可分函数', fontsize=10)
        plt.xlim(bound)
        plt.ylim(bound)
        plt.xticks(fontsize=10, fontfamily='Times New Roman')
        plt.yticks(fontsize=10, fontfamily='Times New Roman')
        plot_contour(cd, bound, bound)
        plt.scatter(xs[i][:, 0], xs[i][:, 1], color='green')
        plt.scatter(means[i][0], means[i][1], color='red')
        plt.xlabel('维度', fontsize=10)
        plt.ylabel('维度', fontsize=10, labelpad=-1)
        plt.savefig(str(i) + '.png', dpi=700, bbox_inches='tight')
        plt.show()


class PlotMaes(MAES):
    def optimize(self, fitness_function=None, args=None):
        fitness = ES.optimize(self, fitness_function)
        z, d, mean, s, tm, y = self.initialize()
        xs, means = [], []  # only for plotting
        while not self._check_terminations():
            z, d, y = self.iterate(z, d, mean, tm, y, args)
            if self.saving_fitness and (not self._n_generations % self.saving_fitness):
                xs.append(self.sigma * d + mean)  # only for plotting
                means.append(mean.copy())  # only for plotting
            mean, s, tm = self._update_distribution(z, d, mean, s, tm, y)
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
        res = self._collect(fitness, y, mean)
        return res, xs, means


if __name__ == '__main__':
    ndim_problem = 2
    problem = {'fitness_function': cd,
               'ndim_problem': ndim_problem,
               'lower_boundary': -10.0 * np.ones((ndim_problem,)),
               'upper_boundary': 10.0 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 3e3,
               'n_individuals': 200,
               'seed_rng': 2022,
               'x': (7.0, -7.0),
               'sigma': 0.05,
               # to record best-so-far fitness every 50 function evaluations
               'saving_fitness': 4,
               'is_restart': False}
    _, xs, means = PlotMaes(problem, options).optimize()
    plot(xs, means)
