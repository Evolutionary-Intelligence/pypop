import os
import numpy as np


# helper functions
def _generate_rotation_matrix(func, ndim, seed):
    """Generate a random rotation matrix of dimension [`ndim` * `ndim`], sampled normally.

        Note that the generated rotation matrix will be automatically stored in txt form for further use.

        :param func: function name, a `str` or `function` object.
        :param ndim: number of dimensions of the rotation matrix, an `int` scalar.
        :param seed: seed for random number generator, a `int` scalar.
        :return: rotation matrix, a [`ndim` * `ndim`] ndarray.
    """
    if hasattr(func, "__call__"):
        func = func.__name__
    data_folder = "pypop_benchmarks_input_data"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    data_path = os.path.join(data_folder, "rotation_matrix_" + func + "_dim_" + str(ndim) + ".txt")
    rotation_matrix = np.random.default_rng(seed).standard_normal(size=(ndim, ndim))
    for i in range(ndim):
        for j in range(i):
            rotation_matrix[:, i] -= np.dot(rotation_matrix[:, i], rotation_matrix[:, j]) * rotation_matrix[:, j]
        rotation_matrix[:, i] /= np.linalg.norm(rotation_matrix[:, i])
    np.savetxt(data_path, rotation_matrix)
    return rotation_matrix
