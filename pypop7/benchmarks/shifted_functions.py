"""For the online documentation of shifted form of these base functions, refer to:
    https://pypop.readthedocs.io/en/latest/benchmarks.html#shifted-transformed-forms

    Here all of below shifted functions are ordered only in *name*. However,
    in the above online documentation, all of shifted functions are
    classified mainly according to *uni-modality* or *multi-modality*
    (dichotomy).

    Note that sometimes the *shifted* form is also called the *transformed* form.
"""
import os

import numpy as np

from pypop7.benchmarks import base_functions as bf
from pypop7.benchmarks.base_functions import squeeze_and_check


# helper functions
def generate_shift_vector(func, ndim, low, high, seed=None):
    """Generate a *random* shift vector of dimension `ndim`, sampled uniformly
       between `low` (inclusive) and `high` (exclusive).

       .. note:: The generated shift vector will be automatically stored in
          the *txt* form **for further use**.

    Parameters
    ----------
    func : str or func
           function name.
    ndim : int
           number of dimensions of the shift vector.
    low  : float or array_like
           lower boundary of the shift vector.
    high : float or array_like
           upper boundary of the shift vector.
    seed : int
           a scalar seed for random number generator (RNG).

    Returns
    -------
    shift_vector : ndarray (of dtype np.float64)
                   a shift vector of size `ndim` sampled uniformly in [`low`, `high`).
    """
    low, high = squeeze_and_check(low), squeeze_and_check(high)
    if hasattr(func, '__call__'):
        func = func.__name__
    data_folder = 'pypop_benchmarks_input_data'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    data_path = os.path.join(data_folder, 'shift_vector_' + func + '_dim_' + str(ndim) + '.txt')
    shift_vector = np.random.default_rng(seed).uniform(low, high, size=ndim)
    np.savetxt(data_path, shift_vector)
    return shift_vector


def load_shift_vector(func, x, shift_vector=None):
    """Load the shift vector which needs to be generated in advance.

       .. note:: When `None`, the shift vector should have been generated and stored in *txt* form
          **in advance**.

    Parameters
    ----------
    func         : func
                   function name.
    x            : array_like
                   decision vector.
    shift_vector : array_like
                   a shift vector with the same size as `x`.

    Returns
    -------
    shift_vector : ndarray (of dtype np.float64)
                   a shift vector with the same size as `x`.
    """
    x = squeeze_and_check(x)
    if shift_vector is None:
        if (not hasattr(func, 'pypop_shift_vector')) or (func.pypop_shift_vector.size != x.size):
            data_folder = 'pypop_benchmarks_input_data'
            data_path = os.path.join(data_folder, 'shift_vector_' + func.__name__ + '_dim_' + str(x.size) + '.txt')
            shift_vector = np.loadtxt(data_path)
            func.pypop_shift_vector = shift_vector
        shift_vector = func.pypop_shift_vector
    shift_vector = squeeze_and_check(shift_vector)
    if shift_vector.shape != x.shape:
        raise TypeError('shift_vector should have the same shape as x.')
    if shift_vector.size != x.size:
        raise TypeError('shift_vector should have the same size as x.')
    return shift_vector


def sphere(x, shift_vector=None):
    """**Sphere** test function.

       .. note:: It's LaTeX formulation is `$\sum_{i=1}^{n}x_i^2$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(sphere, x, shift_vector)
    y = base_functions.sphere(x - shift_vector)
    return y


def cigar(x, shift_vector=None):
    """**Cigar** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(cigar, x, shift_vector)
    y = base_functions.cigar(x - shift_vector)
    return y


def discus(x, shift_vector=None):  # also called tablet
    """**Discus** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(discus, x, shift_vector)
    y = base_functions.discus(x - shift_vector)
    return y


def cigar_discus(x, shift_vector=None):
    """**Cigar-Discus** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(cigar_discus, x, shift_vector)
    y = base_functions.cigar_discus(x - shift_vector)
    return y


def ellipsoid(x, shift_vector=None):
    """**Ellipsoid** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(ellipsoid, x, shift_vector)
    y = base_functions.ellipsoid(x - shift_vector)
    return y


def different_powers(x, shift_vector=None):
    """**Different-Powers** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(different_powers, x, shift_vector)
    y = base_functions.different_powers(x - shift_vector)
    return y


def schwefel221(x, shift_vector=None):
    """**Schwefel221** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(schwefel221, x, shift_vector)
    y = base_functions.schwefel221(x - shift_vector)
    return y


def step(x, shift_vector=None):
    """**Step** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(step, x, shift_vector)
    y = base_functions.step(x - shift_vector)
    return y


def schwefel222(x, shift_vector=None):
    """**Schwefel222** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(schwefel222, x, shift_vector)
    y = base_functions.schwefel222(x - shift_vector)
    return y


def rosenbrock(x, shift_vector=None):
    """**Rosenbrock** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(rosenbrock, x, shift_vector)
    y = base_functions.rosenbrock(x - shift_vector)
    return y


def schwefel12(x, shift_vector=None):
    """**Schwefel12** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(schwefel12, x, shift_vector)
    y = base_functions.schwefel12(x - shift_vector)
    return y


def exponential(x, shift_vector=None):
    """**Exponential** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(exponential, x, shift_vector)
    y = base_functions.exponential(x - shift_vector)
    return y


def griewank(x, shift_vector=None):
    """**Griewank** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(griewank, x, shift_vector)
    y = base_functions.griewank(x - shift_vector)
    return y


def bohachevsky(x, shift_vector=None):
    """**Bohachevsky** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(bohachevsky, x, shift_vector)
    y = base_functions.bohachevsky(x - shift_vector)
    return y


def ackley(x, shift_vector=None):
    """**Ackley** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(ackley, x, shift_vector)
    y = base_functions.ackley(x - shift_vector)
    return y


def rastrigin(x, shift_vector=None):
    """**Rastrigin** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(rastrigin, x, shift_vector)
    y = base_functions.rastrigin(x - shift_vector)
    return y


def scaled_rastrigin(x, shift_vector=None):
    """**Scaled-Rastrigin** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(scaled_rastrigin, x, shift_vector)
    y = base_functions.scaled_rastrigin(x - shift_vector)
    return y


def skew_rastrigin(x, shift_vector=None):
    """**Skew-Rastrigin** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(skew_rastrigin, x, shift_vector)
    y = base_functions.skew_rastrigin(x - shift_vector)
    return y


def levy_montalvo(x, shift_vector=None):
    """**Levy-Montalvo** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(levy_montalvo, x, shift_vector)
    y = base_functions.levy_montalvo(x - shift_vector)
    return y


def michalewicz(x, shift_vector=None):
    """**Michalewicz** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(michalewicz, x, shift_vector)
    y = base_functions.michalewicz(x - shift_vector)
    return y


def salomon(x, shift_vector=None):
    """**Salomon** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(salomon, x, shift_vector)
    y = base_functions.salomon(x - shift_vector)
    return y


def shubert(x, shift_vector=None):
    """**Shubert** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(shubert, x, shift_vector)
    y = base_functions.shubert(x - shift_vector)
    return y


def schaffer(x, shift_vector=None):
    """**Schaffert** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `shift_vector` is `None`, please use function `generate_shift_vector()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x            : ndarray
                   input vector.
    shift_vector : ndarray
                   a vector with the same size as `x`.

    Returns
    -------
    y : float
        scalar fitness.
    """
    shift_vector = load_shift_vector(schaffer, x, shift_vector)
    y = base_functions.schaffer(x - shift_vector)
    return y
