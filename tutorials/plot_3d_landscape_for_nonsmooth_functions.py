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


if __name__ == '__main__':
    bound = [-10.0, 10.0]
    fig = plt.figure(figsize=(2.5, 2.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    x, y, z = generate_xyz(schwefel221, bound, bound, 1000)
    ax.plot_surface(x, y, z, cmap='cool',
                    rstride=100, cstride=100)
    ax.set_title('非平滑函数', fontsize=10)
    ax.set_xlim(bound)
    ax.set_ylim(bound)
    ax.set_xticks([-10.0, -5.0, 0.0, 5.0, 10.0], fontsize=10)
    ax.set_yticks([-10.0, -5.0, 0.0, 5.0, 10.0], fontsize=10)
    ax.set_zticks([0.0, 5.0, 10.0], fontsize=10)
    ax.set_xlabel('维度1', fontsize=10)
    ax.set_ylabel('维度2', fontsize=10)
    ax.set_zlabel('适应值', fontsize=10, rotation='vertical')
    plt.savefig('3d_schwefel221.png', dpi=700, bbox_inches='tight')
    plt.show()
