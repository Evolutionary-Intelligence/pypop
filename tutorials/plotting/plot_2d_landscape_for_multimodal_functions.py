"""This script has been used in Qiqi Duan's Ph.D. Dissertation (HIT&SUSTech).

    Chinese: 该绘图脚本被段琦琦的博士论文（哈工大与南科大联合培养）所使用。
"""
import matplotlib
import matplotlib.pyplot as plt

from pypop7.benchmarks.utils import generate_xyz
from pypop7.benchmarks.base_functions import ackley
from pypop7.benchmarks.base_functions import griewank


font_size = 10
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = font_size  # 对应5号字体


if __name__ == '__main__':
    ndim_problem = 2
    bound=[-10.0, 10.0]
    plt.figure(figsize=(2.5, 2.5))
    plt.title('多峰函数', fontsize=font_size)
    plt.xlim(bound)
    plt.ylim(bound)
    plt.xticks(fontsize=font_size, fontfamily='Times New Roman')
    plt.yticks(fontsize=font_size, fontfamily='Times New Roman')
    x, y, z = generate_xyz(ackley, bound, bound, 500)
    plt.contourf(x, y, z, cmap='Blues')
    plt.contour(x, y, z, colors='white')
    plt.imshow(z, cmap='Blues', interpolation='none')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=10)
    for lf in cbar.ax.yaxis.get_ticklabels():
        lf.set_family('Times New Roman')
    plt.xlabel('维度', fontsize=font_size)
    plt.ylabel('维度', fontsize=font_size, labelpad=-1)
    plt.savefig(str(ackley.__name__) + '.png', dpi=700, bbox_inches='tight')
    plt.show()

    ndim_problem = 2
    bound=[-10.0, 10.0]
    plt.figure(figsize=(2.5, 2.5))
    plt.title('多峰函数', fontsize=font_size)
    plt.xlim(bound)
    plt.ylim(bound)
    plt.xticks(fontsize=font_size, fontfamily='Times New Roman')
    plt.yticks(fontsize=font_size, fontfamily='Times New Roman')
    x, y, z = generate_xyz(griewank, bound, bound, 500)
    plt.contourf(x, y, z, cmap='Blues')
    plt.contour(x, y, z, colors='white')
    plt.imshow(z, cmap='Blues', interpolation='none')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=10)
    for lf in cbar.ax.yaxis.get_ticklabels():
        lf.set_family('Times New Roman')
    plt.xlabel('维度', fontsize=font_size)
    plt.ylabel('维度', fontsize=font_size, labelpad=-1)
    plt.savefig(str(griewank.__name__) + '.png', dpi=700, bbox_inches='tight')
    plt.show()
