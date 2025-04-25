"""This script has been used in Qiqi Duan's Ph.D. Dissertation (HIT&SUSTech).

    Chinese: 该绘图脚本被段琦琦的博士论文（哈工大与南科大联合培养）所使用。
"""
import matplotlib
import matplotlib.pyplot as plt

from pypop7.benchmarks.utils import generate_xyz
from pypop7.benchmarks.base_functions import schwefel221


font_size = 10
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = font_size  # 对应5号字体


if __name__ == '__main__':
    ndim_problem = 2
    bound=[-10.0, 10.0]
    plt.figure(figsize=(2.5, 2.5))
    plt.title('非平滑函数', fontsize=10)
    plt.xlim(bound)
    plt.ylim(bound)
    plt.xticks(fontsize=font_size, fontfamily='Times New Roman')
    plt.yticks(fontsize=font_size, fontfamily='Times New Roman')
    x, y, z = generate_xyz(schwefel221, bound, bound, 300)
    plt.contourf(x, y, z, cmap='Blues')
    plt.contour(x, y, z, colors='white')
    cax = plt.imshow(z, cmap='Blues', interpolation='none')
    cbar = plt.colorbar(cax)
    cbar.ax.tick_params(labelsize=10)
    for lf in cbar.ax.yaxis.get_ticklabels():
        lf.set_family('Times New Roman')
    plt.xlabel('维度', fontsize=font_size)
    plt.ylabel('维度', fontsize=font_size, labelpad=-1)
    plt.savefig(str(schwefel221.__name__) + '.png',
                dpi=700, bbox_inches='tight')
    plt.show()
