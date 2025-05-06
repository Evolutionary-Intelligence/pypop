"""Compare computational efficiency between
    *naive* and *NumPy* matrix multiplication.

    Chinese: 该绘图脚本被段琦琦的博士论文（哈工大与南科大联合培养）所使用。
"""
import time

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def naive_matrix_dot(a, b, md):
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                md[i][j] += a[i][k] * b[k][j]


if __name__ == '__main__':
    size = 500
    a, b= np.random.rand(size, size), np.random.rand(size, size)
    md_naive, md_np = np.zeros((size, size)), np.zeros((size, size))

    start_time = time.time()
    naive_matrix_dot(a, b, md_naive)
    end_time = time.time()
    runtime_naive = end_time - start_time
    print(runtime_naive)

    start_time = time.time()
    md_np = np.dot(a, b)
    end_time = time.time()
    runtime_np = end_time - start_time
    print(runtime_np)

    print('Speedup: {:2}'.format(runtime_naive / runtime_np))
    print(np.all(np.abs(md_naive - md_np) < 1e-12))

    sns.set_theme(style='dark')
    fig = plt.figure(figsize=(2.2, 2.2))
    plt.rcParams['font.size'] = '10'
    plt.rcParams['font.family'] = 'Times New Roman'
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    xticks = []
    algos = ['naive', 'numpy']
    runtime = [runtime_naive, runtime_np]
    colors = ["#F08C55", "#6EC8C8"]
    for j, a in enumerate(algos):
        ax1.bar([0.5 + j], [runtime[j]], fc=colors[j])
        xticks.append(0.5 + j)
    ax1.set_ylabel('Runtime')
    ax1.set_xticks(xticks, algos)
    # ax1.set_yticks([, , ], ['', '', ''])
    ax1.set_yscale('log')
    ax1.set_xlim(0, len(xticks))
    ax2.plot(np.ones(len(xticks) + 1,) * runtime_naive / runtime_np)
    ax2.tick_params(colors='m')
    ax2.set_ylabel('Speedup', color='m', labelpad=-1)
    ax2.set_yticks([1.7e4, 1.9e4, 2.1e4],
                   ['1.7e4', '1.9e4', '2.1e4'])
    plt.title('Naive vs NumPy')
    plt.savefig('compare_matrix_dot.png',
                dpi=700, bbox_inches='tight')
    plt.show()
