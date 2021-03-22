import os
import numpy as np

from benchmarks import base_functions
from benchmarks.base_functions import _squeeze_and_check


# helper functions
def _generate_rotation_matrix(func, ndim, seed):
    """Generate a random rotation matrix of dimension [`ndim` * `ndim`], sampled normally.

        Note that the generated rotation matrix will be automatically stored in txt form for further use.

    :param func: function name, a `str` or `function` object.
    :param ndim: number of dimensions of the rotation matrix, an `int` scalar.
    :param seed: seed for random number generator, a `int` scalar.
    :return: rotation matrix, a [`ndim` * `ndim`] ndarray.
    """
    if hasattr(func, '__call__'):
        func = func.__name__
    data_folder = 'pypop_benchmarks_input_data'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    data_path = os.path.join(data_folder, 'rotation_matrix_' + func + '_dim_' + str(ndim) + '.txt')
    rotation_matrix = np.random.default_rng(seed).standard_normal(size=(ndim, ndim))
    for i in range(ndim):
        for j in range(i):
            rotation_matrix[:, i] -= np.dot(rotation_matrix[:, i], rotation_matrix[:, j]) * rotation_matrix[:, j]
        rotation_matrix[:, i] /= np.linalg.norm(rotation_matrix[:, i])
    np.savetxt(data_path, rotation_matrix)
    return rotation_matrix


def _load_rotation_matrix(func, x, rotation_matrix=None):
    """Load the rotation matrix which needs to be generated in advance.
        When `None`, the rotation matrix should have been generated and stored in txt form in advance.

    :param func: function name, a `function` object.
    :param x: decision vector, array_like of floats.
    :param rotation_matrix: rotation matrix, array_like of floats.
    :return: rotation matrix, a 2-d `ndarray` of `dtype` `np.float64`, whose shape is `(x.size, x.size)`.
    """
    x = _squeeze_and_check(x)
    if rotation_matrix is None:
        if (not hasattr(func, 'pypop_rotation_matrix')) or (func.pypop_rotation_matrix.shape != (x.size, x.size)):
            data_folder = 'pypop_benchmarks_input_data'
            data_path = os.path.join(data_folder, 'rotation_matrix_' + func.__name__ + '_dim_' + str(x.size) + '.txt')
            rotation_matrix = np.loadtxt(data_path)
            if rotation_matrix.size == 1:
                rotation_matrix = np.array([[float(rotation_matrix)]])
            func.pypop_rotation_matrix = rotation_matrix
        rotation_matrix = func.pypop_rotation_matrix
    if rotation_matrix.shape != (x.size, x.size):
        raise TypeError(f'rotation matrix should have shape: {(x.size, x.size)}.')
    return rotation_matrix


def sphere(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(sphere, x, rotation_matrix)
    y = base_functions.sphere(np.dot(rotation_matrix, x))
    return y


def cigar(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(cigar, x, rotation_matrix)
    y = base_functions.cigar(np.dot(rotation_matrix, x))
    return y


def discus(x, rotation_matrix=None):  # also called tablet
    rotation_matrix = _load_rotation_matrix(discus, x, rotation_matrix)
    y = base_functions.discus(np.dot(rotation_matrix, x))
    return y


def cigar_discus(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(cigar_discus, x, rotation_matrix)
    y = base_functions.cigar_discus(np.dot(rotation_matrix, x))
    return y


def ellipsoid(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(ellipsoid, x, rotation_matrix)
    y = base_functions.ellipsoid(np.dot(rotation_matrix, x))
    return y


def different_powers(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(different_powers, x, rotation_matrix)
    y = base_functions.different_powers(np.dot(rotation_matrix, x))
    return y


def schwefel221(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(schwefel221, x, rotation_matrix)
    y = base_functions.schwefel221(np.dot(rotation_matrix, x))
    return y


def rosenbrock(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(rosenbrock, x, rotation_matrix)
    y = base_functions.rosenbrock(np.dot(rotation_matrix, x))
    return y


def schwefel12(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(schwefel12, x, rotation_matrix)
    y = base_functions.schwefel12(np.dot(rotation_matrix, x))
    return y
