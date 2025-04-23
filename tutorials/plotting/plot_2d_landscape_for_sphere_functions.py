"""This script has been used in Qiqi Duan's Ph.D. Dissertation (HIT&SUSTech).

    Chinese: 该绘图脚本被段琦琦的博士论文（哈工大与南科大联合培养）所使用。
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import sphere
from pypop7.benchmarks.utils import generate_xyz


font_size = 10
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = font_size  # 对应5号字体


# helper function for 2D-plotting
def plot_contour(func, x, y):
    levels = [sphere(i * np.ones((2,))) for i in np.arange(0, 12, 2)]
    x, y, z = generate_xyz(func, x, y, 500)
    plt.contourf(x, y, z, cmap='cool', levels=levels)
    plt.contour(x, y, z, colors='white', levels=levels)


if __name__ == '__main__':
    bound=[-10.0, 10.0]
    plt.figure(figsize=(2.5, 2.5))
    plt.title('简单函数', fontsize=font_size)
    plt.xlim(bound)
    plt.ylim(bound)
    plt.xticks(fontsize=font_size, fontfamily='Times New Roman')
    plt.yticks(fontsize=font_size, fontfamily='Times New Roman')
    plot_contour(sphere, bound, bound)
    plt.xlabel('维度', fontsize=font_size)
    plt.ylabel('维度', fontsize=font_size, labelpad=-1)
    plt.savefig(str(sphere.__name__) + '.png',
                dpi=700, bbox_inches='tight')
    plt.show()
