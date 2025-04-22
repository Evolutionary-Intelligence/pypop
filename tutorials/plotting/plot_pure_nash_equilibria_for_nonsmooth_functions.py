"""This script has been used in Qiqi Duan's Ph.D. Dissertation (HIT&SUSTech).

    Chinese: 该绘图脚本被段琦琦的博士论文（哈工大与南科大联合培养）所使用。
"""
import matplotlib
import matplotlib.pyplot as plt

from pypop7.benchmarks.utils import generate_xyz


matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10  # 对应5号字体


def non_smooth(x):
    return abs(x[0] - x[1]) - min(x)


# helper function for 2D-plotting
def plot_contour(func, x, y):
    x, y, z = generate_xyz(func, x, y, 500)
    plt.contourf(x, y, z, cmap='bone')
    plt.contour(x, y, z, colors='white')


if __name__ == '__main__':
    ndim_problem = 2
    bound=[-10.0, 10.0]
    plt.figure(figsize=(2.5, 2.5))
    plt.xlim(bound)
    plt.ylim(bound)
    plt.xticks(fontsize=10, fontfamily='Times New Roman')
    plt.yticks(fontsize=10, fontfamily='Times New Roman')
    plot_contour(non_smooth, bound, bound)
    plt.xlabel('维度', fontsize=10)
    plt.ylabel('维度', fontsize=10, labelpad=-1)
    plt.plot([-10.0, 10.0], [-10.0, 10.0], c='green', linewidth=3)
    plt.savefig(str(non_smooth.__name__) + '.png', dpi=700, bbox_inches='tight')
    plt.show()
