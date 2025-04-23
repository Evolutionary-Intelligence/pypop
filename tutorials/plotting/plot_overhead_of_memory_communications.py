"""This script has been used in Qiqi Duan's Ph.D. Dissertation (HIT&SUSTech).

    Chinese: 该绘图脚本被段琦琦的博士论文（哈工大与南科大联合培养）所使用。
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


n = 2000  # problem dimensionality
# a set of number of parallel communications
p = np.linspace(100, 600, 60)

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10  # 对应5号字体

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot()
font_size = 10

for m in [2000, 1000, 500, 100]:  # number of evolution paths
    y = (p * n * m * 8) / (1024 ** 3)  # GB
    ax.plot(p, y, label=f'{m}',
            linewidth=4, markersize=14)

ax.set_xlabel('通信数量', fontsize=font_size)
ax.set_ylabel('内存需求量', fontsize=font_size)
ax.set_title("通信交流成本分析", fontsize=font_size)
font_prop = fm.FontProperties(family='Times New Roman',
                              size='10')
ax.legend(prop=font_prop)
plt.xticks(fontsize=10, fontfamily='Times New Roman')
plt.yticks(fontsize=10, fontfamily='Times New Roman')
plt.savefig('Overhead-Analysis-of-Memory-Communications.png',
            dpi=700, bbox_inches='tight')
plt.show()
