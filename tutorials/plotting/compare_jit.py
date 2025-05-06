"""Compare computational efficiency between
    *NumPy* and *Numba* matrix multiplication.

    Chinese: 该绘图脚本被段琦琦的博士论文（哈工大与南科大联合培养）所使用。
"""
import time

import numpy as np
import numba as nb
import seaborn as sns
import matplotlib.pyplot as plt


def cholesky_update(rm, z, downdate):
    """Cholesky update of rank-one.

    Parameters
    ----------
    rm       : (N, N) ndarray
               2D input data from which the triangular part will be used to
               read the Cholesky factor.
    z        : (N,) ndarray
               1D update/downdate vector.
    downdate : bool
               `False` indicates an update while `True` indicates a downdate (`False` by default).

    Returns
    -------
    D : (N, N) ndarray
        Cholesky factor.
    """
    # https://github.com/scipy/scipy/blob/d20f92fce9f1956bfa65365feeec39621a071932/
    #     scipy/linalg/_decomp_cholesky_update.py
    rm, z, alpha, beta = rm.T, z, np.empty_like(z), np.empty_like(z)
    alpha[-1], beta[-1] = 1.0, 1.0
    sign = -1.0 if downdate else 1.0
    for r in range(len(z)):
        a = z[r] / rm[r, r]
        alpha[r] = alpha[r - 1] + sign * np.power(a, 2)
        beta[r] = np.sqrt(alpha[r])
        z[r + 1:] -= a * rm[r, r + 1:]
        rm[r, r:] *= beta[r] / beta[r - 1]
        rm[r, r + 1:] += sign * a / (beta[r] * beta[r - 1]) * z[r + 1:]
    return rm.T


@nb.jit(nopython=True)
def cholesky_update_jit(rm, z, downdate):
    """Cholesky update of rank-one.

    Parameters
    ----------
    rm       : (N, N) ndarray
               2D input data from which the triangular part will be used to
               read the Cholesky factor.
    z        : (N,) ndarray
               1D update/downdate vector.
    downdate : bool
               `False` indicates an update while `True` indicates a downdate (`False` by default).

    Returns
    -------
    D : (N, N) ndarray
        Cholesky factor.
    """
    # https://github.com/scipy/scipy/blob/d20f92fce9f1956bfa65365feeec39621a071932/
    #     scipy/linalg/_decomp_cholesky_update.py
    rm, z, alpha, beta = rm.T, z, np.empty_like(z), np.empty_like(z)
    alpha[-1], beta[-1] = 1.0, 1.0
    sign = -1.0 if downdate else 1.0
    for r in range(len(z)):
        a = z[r] / rm[r, r]
        alpha[r] = alpha[r - 1] + sign * np.power(a, 2)
        beta[r] = np.sqrt(alpha[r])
        z[r + 1:] -= a * rm[r, r + 1:]
        rm[r, r:] *= beta[r] / beta[r - 1]
        rm[r, r + 1:] += sign * a / (beta[r] * beta[r - 1]) * z[r + 1:]
    return rm.T


if __name__ == '__main__':
    size = 2000
    rm, z = np.random.rand(size, size), np.random.rand(size)
    rm_numpy, rm_numba = np.zeros((size, size)), np.zeros((size, size))

    start_time = time.time()
    for i in range(5000):
        rm_numba = cholesky_update_jit(rm, z, False)
    end_time = time.time()
    runtime_nb = end_time - start_time
    print(runtime_nb)

    start_time = time.time()
    for i in range(5000):
        rm_numpy = cholesky_update(rm, z, False)
    end_time = time.time()
    runtime_np = end_time - start_time
    print(runtime_np)

    print('Speedup: {:2}'.format(runtime_np / runtime_nb))
    print(np.all(np.abs(rm_numba - rm_numpy) < 1e-6))

    sns.set_theme(style='dark')
    fig = plt.figure(figsize=(2.2, 2.2))
    plt.rcParams['font.size'] = '10'
    plt.rcParams['font.family'] = 'Times New Roman'
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    xticks = []
    algos = ['numba', 'numpy']
    runtime = [runtime_nb, runtime_np]
    colors = ["#F08C55", "#6EC8C8"]
    for j, a in enumerate(algos):
        ax1.bar([0.5 + j], [runtime[j]], fc=colors[j])
        xticks.append(0.5 + j)
    ax1.set_ylabel('Runtime')
    ax1.set_xticks(xticks, algos)
    # ax1.set_yticks([, , ], ['', '', ''])
    ax1.set_yscale('log')
    ax1.set_xlim(0, len(xticks))
    ax2.plot(np.ones(len(xticks) + 1,) * runtime_np / runtime_nb)
    ax2.tick_params(colors='m')
    ax2.set_ylabel('Speedup', color='m', labelpad=-1)
    # ax2.set_yticks([], [])
    plt.title('Naive vs NumPy')
    plt.savefig('compare_matrix_dot.png',
                dpi=700, bbox_inches='tight')
    plt.show()
