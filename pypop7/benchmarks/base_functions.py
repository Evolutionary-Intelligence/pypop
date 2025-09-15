"""Online documentation:
    https://pypop.readthedocs.io/en/latest/benchmarks.html#base-functions
"""
import math

import numpy as np


# helper function
def squeeze_and_check(x, size_gt_1=False):
    """Squeeze the input `x` into 1-d `numpy.ndarray` and check
        whether its number of dimensions == 1. If not, raise a TypeError.
        Optionally, check whether its size > 1. If not, raise a TypeError.
    """
    x = np.squeeze(x)
    if (x.ndim == 0) and (x.size == 1):
        x = np.array([x])
    if x.ndim != 1:
        raise TypeError(f'The number of dimensions should == 1 (not {x.ndim}) after numpy.squeeze(x).')
    if size_gt_1 and not (x.size > 1):
        raise TypeError(f'The size should > 1 (not {x.size}) after numpy.squeeze(x).')
    if x.size == 0:
        raise TypeError(f'the size should != 0.')
    return x


def cigar(x):
    """**Cigar** test function.

       .. note:: Its dimensionality should `> 1`.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    x = squeeze_and_check(x, True)
    x = np.square(x)
    y = x[0] + (10.0 ** 6) * np.sum(x[1:])
    return y


def cigar_discus(x):
    """**Cigar-Discus** test function.

       .. note:: Its dimensionality should `> 1`.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    x = squeeze_and_check(x, True)
    x = np.square(x)
    if x.size == 2:
        y = x[0] + (10.0 ** 4) * np.sum(x) + (10.0 ** 6) * x[-1]
    else:
        y = x[0] + (10.0 ** 4) * np.sum(x[1:-1]) + (10.0 ** 6) * x[-1]
    return y


def different_powers(x):
    """**Different-Powers** test function.

       .. note:: Its dimensionality should `> 1`.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    x = squeeze_and_check(x, True)
    w = 2.0 + 4.0 * np.linspace(0.0, 1.0, len(x))
    y = np.sum(np.power(np.abs(x), w))
    return y


def discus(x):
    """**Discus** (also called **Tablet**) test function.

       .. note:: Its dimensionality should `> 1`.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    x = squeeze_and_check(x, True)
    x = np.square(x)
    y = (10.0 ** 6) * x[0] + np.sum(x[1:])
    return y


def ellipsoid(x):
    """**Ellipsoid** test function.

       .. note:: Its dimensionality should `> 1`.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    x = squeeze_and_check(x, True)
    w = 6.0 * np.linspace(0.0, 1.0, len(x))
    y = np.dot(np.power(10.0, w), np.square(x))
    return y


def exponential(x):
    """**Exponential** test function.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    x = squeeze_and_check(x)
    y = -np.exp(-0.5 * np.sum(np.square(x)))
    return y


def rosenbrock(x):
    """**Rosenbrock** test function.

       .. note:: Its dimensionality should `> 1`.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    x = squeeze_and_check(x, True)
    y = 100.0 * np.sum(np.square(x[1:] - np.square(x[:-1]))) + np.sum(np.square(x[:-1] - 1.0))
    return y


def schwefel12(x):
    """**Schwefel12** test function.

       .. note:: Its dimensionality should `> 1`.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    x = squeeze_and_check(x, True)
    x = [np.sum(x[:i + 1]) for i in range(len(x))]
    y = np.sum(np.square(x))
    return y


def schwefel221(x):
    """**Schwefel221** test function.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    y = np.max(np.abs(squeeze_and_check(x)))
    return y


def schwefel222(x):
    """**Schwefel222** test function.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    x = np.abs(squeeze_and_check(x))
    y = np.sum(x) + np.prod(x)
    return y


def sphere(x):
    """**Sphere** test function.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    y = np.sum(np.square(squeeze_and_check(x)))
    return y


def step(x):
    """**Step** test function.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    y = np.sum(np.square(np.floor(squeeze_and_check(x) + 0.5)))
    return y


def griewank(x):
    """**Griewank** test function.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x = squeeze_and_check(x)
    y = np.sum(np.square(x)) / 4000.0 - np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1)))) + 1.0
    return y


class Griewank(BaseFunction):
    def __call__(self, x):
        """Class of **Griewank** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return griewank(x)


def bohachevsky(x):
    """**Bohachevsky** test function.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x, y = squeeze_and_check(x), 0.0
    for i in range(x.size - 1):
        y += np.square(x[i]) + 2.0 * np.square(x[i + 1]) - 0.3 * np.cos(3.0 * np.pi * x[i]) - \
             0.4 * np.cos(4.0 * np.pi * x[i + 1]) + 0.7
    return y


class Bohachevsky(BaseFunction):
    def __call__(self, x):
        """

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return bohachevsky(x)


def ackley(x):
    """**Ackley** test function.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x = squeeze_and_check(x)
    y = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(np.square(x)) / x.size)) - \
        np.exp(np.sum(np.cos(2.0 * np.pi * x)) / x.size) + 20.0 + np.exp(1)
    return y


class Ackley(BaseFunction):
    def __call__(self, x):
        """Class of **Ackley** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return ackley(x)


def rastrigin(x):
    """**Rastrigin** test function.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    x = squeeze_and_check(x)
    y = 10.0 * len(x) + np.sum(np.square(x) - 10.0 * np.cos(2.0 * np.pi * x))
    return y


class Rastrigin(BaseFunction):
    def __call__(self, x):
        """

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return rastrigin(x)


def scaled_rastrigin(x):
    """**Scaled-Rastrigin** test function.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    w = np.power(10.0, np.linspace(0.0, 1.0, len(x)))
    y = rastrigin(squeeze_and_check(x) * w)
    return y


class ScaledRastrigin(BaseFunction):
    def __call__(self, x):
        """Class of **Scaled-Rastrigin** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return scaled_rastrigin(x)


def skew_rastrigin(x):
    """**Skew-Rastrigin** test function.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x = squeeze_and_check(x)
    for i in range(x.size):
        if x[i] > 0.0:
            x[i] *= 10.0
    y = rastrigin(x)
    return y


class SkewRastrigin(BaseFunction):
    def __call__(self, x):
        """Class of **Skew-Rastrigin** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return skew_rastrigin(x)


def levy_montalvo(x):
    """**Levy-Montalvo** test function.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x, y = 1.0 + 0.25 * (squeeze_and_check(x) + 1.0), 0.0
    for i in range(x.size - 1):
        y += np.square(x[i] - 1.0) * (1.0 + 10.0 * np.square(np.sin(np.pi * x[i + 1])))
    y += 10.0 * np.square(np.sin(np.pi * x[0])) + np.square(x[-1] - 1.0)
    return (np.pi / x.size) * y


class LevyMontalvo(BaseFunction):
    def __call__(self, x):
        """Class of **Levy-Montalvo** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return levy_montalvo(x)


def michalewicz(x):
    """**Michalewicz** test function.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    x, y = squeeze_and_check(x), 0.0
    for i in range(x.size):
        y -= np.sin(x[i]) * np.power(np.sin((i + 1) * np.square(x[i]) / np.pi), 20.0)
    return y


class Michalewicz(BaseFunction):
    def __call__(self, x):
        """Class of **Michalewicz** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return michalewicz(x)


def salomon(x):
    """**Salomon** test function.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    x = np.linalg.vector_norm(squeeze_and_check(x))
    y = 1.0 - np.cos(2.0 * np.pi * x) + 0.1 * x
    return y


class Salomon(BaseFunction):
    def __call__(self, x):
        """Class of **Salomon** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return salomon(x)


def shubert(x):
    """**Shubert** test function.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x, y = squeeze_and_check(x), 1.0
    for i in range(x.size):
        yy = 0.0
        for j in range(1, 6):
            yy += j * np.cos((j + 1) * x[i] + j)
        y *= yy
    return y


class Shubert(BaseFunction):
    def __call__(self, x):
        """Class of **Shubert** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return shubert(x)


def schaffer(x):
    """**Schaffer** test function.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    x, y = squeeze_and_check(x), 0.0
    for i in range(x.size - 1):
        xx = np.power(x[i], 2.0) + np.power(x[i + 1], 2.0)
        y += np.power(xx, 0.25) * (np.power(np.sin(50.0 * np.power(xx, 0.1)), 2.0) + 1.0)
    return y


class Schaffer(BaseFunction):
    def __call__(self, x):
        """Class of **Schaffer** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return schaffer(x)
