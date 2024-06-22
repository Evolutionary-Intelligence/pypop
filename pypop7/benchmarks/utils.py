import numpy as np  # engine for numerical computing
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt


sns.set_theme(style='darkgrid')


# helper function for 2D-plotting
def generate_xyz(func, x, y, num=200):
    """Generate necessary data before plotting a 2D contour of the fitness landscape.

    Parameters
    ----------
    func : func
           benchmarking function.
    x    : list
           x-axis range.
    y    : list
           y-axis range.
    num  : int
           number of samples in each of x- and y-axis range.

    Returns
    -------
    tuple
        A (x, y, z) tuple where x, y, and z are data points in
        x-axis, y-axis, and function values, respectively.
    """
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


# helper function for 2D-plotting
def plot_contour(func, x, y, levels=None, num=200, is_save=False):
    """Plot a 2D contour of the fitness landscape.

    Parameters
    ----------
    func    : func
              benchmarking function.
    x       : list
              x-axis range.
    y       : list
              y-axis range.
    levels  : int
              number of contour lines.
    num     : int
              number of samples in each of x- and y-axis range.
    is_save : bool
              whether or not to save the generated figure in the *local* folder.

    Returns
    -------
    An online figure.
    """
    x, y, z = generate_xyz(func, x, y, num)
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


# helper function for 3D-plotting
def plot_surface(func, x, y, num=200, is_save=False):
    """Plot a 3D contour of the fitness landscape.

    Parameters
    ----------
    func    : func
              benchmarking function.
    x       : list
              x-axis range.
    y       : list
              y-axis range.
    num     : int (`200` by default)
              number of samples in each of x- and y-axis range.
    is_save : bool (`False` by default)
              whether or not to save the generated figure in the *local* folder.

    Returns
    -------
    An online figure.
    """
    x, y, z = generate_xyz(func, x, y, num)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(x, y, z, cmap=cm.cool, linewidth=0, antialiased=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')
    plt.title(func.__name__)
    if is_save:
        plt.savefig(func.__name__ + '_surface.png')
    plt.show()
