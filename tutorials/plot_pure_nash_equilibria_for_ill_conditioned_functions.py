"""This script has been used in Qiqi Duan's Ph.D. Dissertation (HIT&SUSTech).

    Chinese: 该绘图脚本被段琦琦的博士论文（哈工大与南科大联合培养）所使用。
"""
import matplotlib
import matplotlib.pyplot as plt

from pypop7.benchmarks.utils import generate_xyz
from pypop7.benchmarks.base_functions import rosenbrock


matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10  # 对应5号字体


def cd(x):  # from https://arxiv.org/pdf/1610.00040v1.pdf
    return 7.0 * (x[0] ** 2) + 6.0 * x[0] * x[1] + 8.0 * (x[1] ** 2)


# helper function for 2D-plotting
def plot_contour(func, x, y):
    x, y, z = generate_xyz(func, x, y, 1000)
    levels = [0, 1, 10, 100, 1000, 10000, 100000, 1000000]
    plt.contourf(x, y, z, cmap='bone', levels=levels)
    plt.contour(x, y, z, colors='white')


if __name__ == '__main__':
    ndim_problem = 2
    bound=[-10.0, 10.0]
    plt.figure(figsize=(2.5, 2.5))
    plt.xlim(bound)
    plt.ylim(bound)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plot_contour(rosenbrock, bound, bound)
    plt.xlabel('维度1', fontsize=10)
    plt.ylabel('维度2', fontsize=10, labelpad=-1)
    plt.scatter([0.0], [0.0], c='green', s=24)
    plt.savefig(str(rosenbrock.__name__) + '.png', dpi=700, bbox_inches='tight')
    plt.show()
