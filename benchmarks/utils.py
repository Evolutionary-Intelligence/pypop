import numpy as np
import matplotlib.pyplot as plt


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


def plot_contour(func, x, y, levels=None, num=200, is_save=False):
    x, y, z = _generate_xyz(func, x, y, num)
    if levels is None:
        plt.contourf(x, y, z, cmap='cool')
        plt.contour(x, y, z, colors='white')
    else:
        plt.contourf(x, y, z, levels, cmap='cool')
        c = plt.contour(x, y, z, levels, colors='white')
        plt.clabel(c, inline=True, fontsize=12, colors='white')
    plt.title(func.__name__)
    plt.xlabel('x')
    plt.ylabel('y')
    if is_save:
        plt.savefig(func.__name__ + '_contour.png')
    plt.show()
