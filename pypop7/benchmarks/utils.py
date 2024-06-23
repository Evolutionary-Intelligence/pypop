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

    Examples
    --------

    .. code-block:: python
       :linenos:

       >>> from pypop7.benchmarks import base_functions
       >>> from pypop7.benchmarks.utils import generate_xyz
       >>> x_, y_, z_ = generate_xyz(base_functions.sphere, [0.0, 1.0], [0.0, 1.0], num=2)
       >>> print(x_.shape, y_.shape, z_.shape)
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

    Examples
    --------

    .. code-block:: python
       :linenos:
       >>> import numpy as np
       >>> from pypop7.benchmarks import rotated_functions as rf
       >>> from pypop7.benchmarks.utils import plot_contour
       >>> plot_contour(bf.sphere, [-10.0, 10.0], [-10.0, 10.0])
       >>> plot_contour(bf.ellipsoid, [-10.0, 10.0], [-10.0, 10.0])
       >>> # plot multi-modality
       >>> plot_contour(bf.rastrigin, [-10.0, 10.0], [-10.0, 10.0])
       >>> # plot non-convexity
       >>> plot_contour(bf.ackley, [-10.0, 10.0], [-10.0, 10.0])
       >>> # plot non-separability
       >>> plot_contour(cd, [-10.0, 10.0], [-10.0, 10.0])
       >>> # plot ill-condition and non-separability
       >>> rf.generate_rotation_matrix(rf.ellipsoid, 2, 72)
       >>> plot_contour(rf.ellipsoid, [-10.0, 10.0], [-10.0, 10.0], 7)

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
    """Plot a 3D surface of the fitness landscape.

    Examples
    --------

    .. code-block:: python
       :linenos:
       >>> import numpy as np
       >>> from pypop7.benchmarks import base_functions as bf
       >>> from pypop7.benchmarks import rotated_functions as rf
       >>> from pypop7.benchmarks.utils import plot_surface
       >>> # plot smoothness and convexity
       >>> plot_surface(bf.sphere, [-10.0, 10.0], [-10.0, 10.0])
       >>> plot_surface(bf.ellipsoid, [-10.0, 10.0], [-10.0, 10.0])
       >>> # plot multi-modality
       >>> plot_surface(bf.rastrigin, [-10.0, 10.0], [-10.0, 10.0])
       >>> # plot non-convexity
       >>> plot_surface(bf.ackley, [-10.0, 10.0], [-10.0, 10.0])
       >>> # plot non-separability
       >>> plot_surface(cd, [-10.0, 10.0], [-10.0, 10.0])
       >>> # plot ill-condition and non-separability
       >>> rf.generate_rotation_matrix(rf.ellipsoid, 2, 72)
       >>> plot_surface(rf.ellipsoid, [-10.0, 10.0], [-10.0, 10.0], 7)

    Parameters
    ----------
    func    : func
              benchmarking function.
    x       : list
              x-axis range.
    y       : list
              y-axis range.
    num     : int
              number of samples in each of x- and y-axis range (`200` by default).
    is_save : bool
              whether or not to save the generated figure in the *local* folder (`False` by default).

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
