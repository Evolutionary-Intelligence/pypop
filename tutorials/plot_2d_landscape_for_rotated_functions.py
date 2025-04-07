"""This script has been used in Qiqi Duan's Ph.D. Dissertation (HIT&SUSTech).

    Chinese: 该绘图脚本被段琦琦的博士论文（哈工大与南科大联合培养）所使用。
"""
import matplotlib
import matplotlib.pyplot as plt

from pypop7.benchmarks.rotated_functions import ellipsoid
from pypop7.benchmarks.rotated_functions import generate_rotation_matrix
from pypop7.benchmarks.utils import generate_xyz


matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10  # 对应5号字体
font_size = 10


# helper function for 2D-plotting
def plot_contour(func, x, y):
    levels = [0, 50000, 20000000, 50000000,
              100000000, 150000000, 200000000]
    generate_rotation_matrix(ellipsoid, 2, 72)
    x, y, z = generate_xyz(func, x, y, 500)
    plt.contourf(x, y, z, cmap='cool', levels=levels)
    plt.contour(x, y, z, colors='white', alpha=0.7)


if __name__ == '__main__':
    bound=[-10.0, 10.0]
    plt.figure(figsize=(2.5, 2.5))
    plt.title('病态函数', fontsize=font_size)
    plt.xlim(bound)
    plt.ylim(bound)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plot_contour(ellipsoid, bound, bound)
    plt.xlabel('维度1', fontsize=font_size)
    plt.ylabel('维度2', fontsize=font_size, labelpad=-1)
    plt.savefig(str(ellipsoid.__name__) + '.png',
                dpi=700, bbox_inches='tight')
    plt.show()
