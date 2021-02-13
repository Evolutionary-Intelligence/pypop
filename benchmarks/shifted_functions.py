import os
import numpy as np

from base_functions import *
from base_functions import _squeeze_and_check


def _generate_shift_vector(func, ndim, low, high, seed=None):
    """Generate a random shift vector of dimension `ndim`, sampled uniformly between
        `low` (inclusive) and `high` (exclusive).

    :param func: function name, a `str` or `function` object.
    :param ndim: number of dimensions of the shift vector, an `int` scalar.
    :param low: lower boundary of the shift vector, a `float` scalar or array_like of floats.
    :param high: upper boundary of the shift vector, a `float` scalar or array_like of floats.
    :param seed: seed for random number generator, a `int` scalar.
    :return: shift_vector: a `ndim`-d vector sampled uniformly in [`low`, `high`).

    https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.uniform.html
    """
    low, high = _squeeze_and_check(low), _squeeze_and_check(high)
    if hasattr(func, "__call__"):
        func = func.__name__
    data_folder = "pypop_benchmarks_input_data"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    data_path = os.path.join(data_folder, "shift_vector_" + func + "_dim_" + str(ndim) + ".txt")
    shift_vector = np.random.default_rng(seed).uniform(low, high, size=ndim)
    np.savetxt(data_path, shift_vector)
    return shift_vector
