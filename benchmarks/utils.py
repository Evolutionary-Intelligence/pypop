import numpy as np


# helper function
def _generate_xyz(func, x, y, num=200):
    x, y = np.array(x), np.array(y)
    if x.size == 2:
        x = np.linspace(x[0], x[1], num)
    if y.size == 2:
        y = np.linspace(y[0], y[1], num)
    x, y = np.meshgrid(x, y)
    z = np.empty(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i, j] = func([x[i, j], y[i, j]])
    return x, y, z
