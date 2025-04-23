"""This script has been used in Qiqi Duan's Ph.D. Dissertation (HIT&SUSTech).

    Chinese: 该绘图脚本被段琦琦的博士论文（哈工大与南科大联合培养）所使用。
"""
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style='darkgrid')
font_size = 10
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = font_size  # 对应5号字体


n = 857  # problem dimensionality
p = ['100', '200', '300']
e = ['25', '440', '857']
s = np.zeros((len(p), len(e)))
for pp, ppp in enumerate(p):
    for ee, eee in enumerate(e):
        s[pp, ee] = (int(ppp) * n * int(eee) * 8) / (1024 ** 3)  # GB
        s[pp, ee] = round(s[pp, ee], 3)
print(s)

width, multiplier = 0.25, 0
fig, ax = plt.subplots(layout='constrained', figsize=(7, 7))

for pi, pp in enumerate(p):
    offset = width * multiplier
    rects = ax.bar(np.arange(len(p)) + offset,
                   s[pi], width)
    ax.bar_label(rects, padding=3,
                 fontsize=font_size, fontfamily='Times New Roman')
    multiplier += 1

ax.set_xlabel('并行总数', fontsize=font_size)
ax.set_ylabel('内存需求量', fontsize=font_size)
ax.set_title("不同设置条件下的通信成本分析", fontsize=font_size)
ax.set_xticks(np.arange(len(p)) + width, p)
ax.set_xticklabels(['100', '200', '300'],
                   fontsize=font_size, fontfamily='Times New Roman')
ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8',
                    '1.0', '1.2', '1.4', '1.6'],
                   fontsize=font_size, fontfamily='Times New Roman')
ax.legend(e, loc='upper left',
          prop={'family': 'Times New Roman', 'size': '10'})
plt.savefig('Overhead-Analysis-of-Memory-Communications.png',
            dpi=700, bbox_inches='tight')
plt.show()
