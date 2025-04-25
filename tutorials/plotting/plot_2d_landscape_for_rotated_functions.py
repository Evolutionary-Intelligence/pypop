"""This script has been used in Qiqi Duan's Ph.D. Dissertation (HIT&SUSTech).

    Chinese: 该绘图脚本被段琦琦的博士论文（哈工大与南科大联合培养）所使用。
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pypop7.benchmarks.rotated_functions import ellipsoid
from pypop7.benchmarks.rotated_functions import generate_rotation_matrix
from pypop7.benchmarks.utils import generate_xyz


font_size = 10
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = font_size  # 对应5号字体


generate_rotation_matrix(ellipsoid, 2, 72)


if __name__ == '__main__':
    bound=[-10.0, 10.0]
    plt.figure(figsize=(2.5, 2.5))
    plt.title('病态函数', fontsize=font_size)
    plt.xlim(bound)
    plt.ylim(bound)
    plt.xticks(fontsize=font_size, fontfamily='Times New Roman')
    plt.yticks(fontsize=font_size, fontfamily='Times New Roman')
    levels = [ellipsoid(i * np.ones((2,))) for i in np.arange(0, 45, 5)]
    x, y, z = generate_xyz(ellipsoid, bound, bound, 500)
    plt.contourf(x, y, z, cmap='Blues', levels=levels)
    plt.contour(x, y, z, colors='white', levels=levels)
    print(np.min(z), np.max(z))
    cax = plt.imshow(z, cmap='Blues', interpolation='none')
    cbar = plt.colorbar(cax)
    cbar.ax.tick_params(labelsize=10)
    for lf in cbar.ax.yaxis.get_ticklabels():
        lf.set_family('Times New Roman')
    plt.xlabel('维度', fontsize=font_size)
    plt.ylabel('维度', fontsize=font_size, labelpad=-1)
    plt.xticks([-10, -5, 0, 5, 10],
               ['-10', '-5', '0', '5', '10'],
               fontsize=font_size,
               fontfamily='Times New Roman')
    plt.yticks([-10, -5, 0, 5, 10],
               ['-10', '-5', '0', '5', '10'],
               fontsize=font_size,
               fontfamily='Times New Roman')
    plt.savefig(str(ellipsoid.__name__) + '.png',
                dpi=700, bbox_inches='tight')
    plt.show()
