import os
import pickle

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

    Parameters
    ----------
    func    : func
              benchmarking function.
    x       : list
              x-axis range.
    y       : list
              y-axis range.
    levels  : int or list
              number of contour lines or a list of contours.
    num     : int
              number of samples in each of x- and y-axis range.
    is_save : bool
              whether or not to save the generated figure in the *local* folder.

    Returns
    -------
    An online figure.

    Examples
    --------

    .. code-block:: python
       :linenos:

       >>> from pypop7.benchmarks.utils import plot_contour
       >>> from pypop7.benchmarks.rotated_functions import generate_rotation_matrix
       >>> from pypop7.benchmarks.rotated_functions import ellipsoid
       >>> # plot ill-condition and non-separability
       >>> generate_rotation_matrix(ellipsoid, 2, 72)
       >>> contour_levels = [0, 5e5, 8e6, 4e7, 8e7, 1.15e8, 1.42e8, 1.62e8, 1.78e8, 1.85e8, 2e8]
       >>> plot_contour(ellipsoid, [-10.0, 10.0], [-10.0, 10.0], contour_levels)
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

    Examples
    --------

    .. code-block:: python
       :linenos:

       >>> from pypop7.benchmarks.utils import plot_surface
       >>> from pypop7.benchmarks.rotated_functions import ellipsoid
       >>> from pypop7.benchmarks.rotated_functions import generate_rotation_matrix
       >>> # plot ill-condition and non-separability
       >>> generate_rotation_matrix(ellipsoid, 2, 72)
       >>> plot_surface(ellipsoid, [-10.0, 10.0], [-10.0, 10.0], 7)
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


# helper function for saving optimization results in pickle form
def save_optimization(results, algo, func, dim, exp, folder='pypop7_benchmarks_lso'):
    """Save optimization results (in **pickle** form) via object serialization.

       .. note:: By default, the **local** file name is given in the following form:
          `Algo-{}_Func-{}_Dim-{}_Exp-{}.pickle`.

    Parameters
    ----------
    results : dict
              optimization results returned by any optimizer.
    algo    : str
              name of algorithm to be used.
    func    : str
              name of the fitness function to be tested.
    dim     : str or int
              dimensionality of the fitness function to be tested.
    exp     : str or int
              index of experiments to be run.
    folder  : str
              local folder under the working space (`pypop7_benchmarks_lso` by default).

    Returns
    -------
    A **local** file stored in the working space (which can be obtained via the `pwd()` command).

    Examples
    --------

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.rs.prs import PRS
       >>> from pypop7.benchmarks.utils import save_optimization
       >>> ndim = 2  # number of dimensionality
       >>> problem = {'fitness_function': rosenbrock,  # to define problem arguments
       ...            'ndim_problem': ndim,
       ...            'lower_boundary': -5.0 * numpy.ones((ndim,)),
       ...            'upper_boundary': 5.0 * numpy.ones((ndim,))}
       >>> options = {'max_function_evaluations': 5000,  # to set optimizer options
       ...            'seed_rng': 2022}  # global step-size may need to be tuned for optimality
       >>> prs = PRS(problem, options)  # to initialize the black-box optimizer class
       >>> res = prs.optimize()  # to run its optimization/evolution process
       >>> save_optimization(res, PRS.__name__, rosenbrock.__name__, ndim, 1)
    """
    if not os.path.exists(folder):
        os.makedirs(folder)  # to make a new folder under the working space
    local_file = os.path.join(folder, 'Algo-{}_Func-{}_Dim-{}_Exp-{}.pickle')  # to set file format
    local_file = local_file.format(str(algo), str(func), str(dim), str(exp))  # to set data format
    with open(local_file, 'wb') as handle:  # to save in pickle form
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def check_optimization(problem, options, results):
    """Check optimization results according to problem arguments and optimizer options.

    Parameters
    ----------
    problem : dict
              problem arguments.
    options : dict
              optimizer options.
    results : dict
              optimization results generated by any black-box optimizer.

    Returns
    -------
    A detailed checking report.

    Examples
    --------

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.utils import check_optimization
       >>> problem = {'lower_boundary': [-5.0, -7.0],
       ...            'upper_boundary': [5.0, 7.0]}
       >>> options = {'max_function_evaluations': 7777777}
       >>> results = {'n_function_evaluations': 7777777,
       ...        'best_so_far_x': np.zeros((2,))}
       >>> check_optimization(problem, options, results)
    """
    # check upper and lower boundary
    if problem.get('lower_boundary') is not None:
        if np.any(results['best_so_far_x'] < np.array(problem.get('lower_boundary'))):
            print("For the best-so-far solution ('best_so_far_x'), " +
                  "there exist some value(s) out of the given 'lower_boundary' in `problem`.")
    if problem.get('upper_boundary') is not None:
        if np.any(results['best_so_far_x'] > np.array(problem.get('upper_boundary'))):
            print("For the best-so-far solution ('best_so_far_x'), " +
                  "there exist some value(s) out of the given 'upper_boundary' in `problem`.")
    # check *max_function_evaluations*
    if options.get('max_function_evaluations') is not None:
        if results['n_function_evaluations'] > options.get('max_function_evaluations'):
            print("The number of function evaluations ('n_function_evaluations') is " +
                  "larger than 'max_function_evaluations' given in `options`.")
