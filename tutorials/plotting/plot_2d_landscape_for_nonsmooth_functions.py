"""This script has been used in Qiqi Duan's Ph.D. Dissertation (HIT&SUSTech).

    Chinese: 该绘图脚本被段琦琦的博士论文（哈工大与南科大联合培养）所使用。
"""
import matplotlib
import matplotlib.pyplot as plt

from pypop7.benchmarks.utils import generate_xyz
from pypop7.benchmarks.base_functions import schwefel221


matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10  # 对应5号字体


# helper function for 2D-plotting
def plot_contour(func, x, y):
    x, y, z = generate_xyz(func, x, y, 500)
    plt.contourf(x, y, z, cmap='cool')
    plt.contour(x, y, z, colors='white')


if __name__ == '__main__':
    ndim_problem = 2
    bound=[-10.0, 10.0]
    plt.figure(figsize=(2.5, 2.5))
    plt.title('非平滑函数', fontsize=10)
    plt.xlim(bound)
    plt.ylim(bound)
    plt.xticks(fontsize=10, fontfamily='Times New Roman')
    plt.yticks(fontsize=10, fontfamily='Times New Roman')
    plot_contour(schwefel221, bound, bound)
    plt.xlabel('维度', fontsize=10)
    plt.ylabel('维度', fontsize=10, labelpad=-1)
    plt.savefig(str(schwefel221.__name__) + '.png',
                dpi=700, bbox_inches='tight')
    plt.show()
