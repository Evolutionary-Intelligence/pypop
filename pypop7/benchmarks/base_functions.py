"""Online documentation:
    https://pypop.readthedocs.io/en/latest/benchmarks.html#base-functions
"""
import math

import numpy as np  # engine for numerical computing


# helper function
def squeeze_and_check(x, size_gt_1=False):
    """Squeeze the input `x` into 1-d `numpy.ndarray`.
        And check whether its number of dimensions == 1. If not, raise a TypeError.
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


# helper class
class BaseFunction(object):
    """Class for all base functions.
    """

    def __init__(self):
        pass


def sphere(x):
    """**Sphere** test function.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    y = np.sum(np.square(squeeze_and_check(x)))
    return y


class Sphere(BaseFunction):
    def __call__(self, x):
        """Class of **Sphere** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return sphere(x)


def cigar(x):
    """**Cigar** test function.

       .. note:: Its dimensionality should `> 1`.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x = np.square(squeeze_and_check(x, True))
    y = x[0] + (10.0 ** 6) * np.sum(x[1:])
    return y


class Cigar(BaseFunction):
    def __call__(self, x):
        """Class of **Cigar** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return cigar(x)


def discus(x):
    """**Discus** (also called **Tablet**) test function.

       .. note:: Its dimensionality should `> 1`.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x = np.square(squeeze_and_check(x, True))
    y = (10.0 ** 6) * x[0] + np.sum(x[1:])
    return y


class Discus(BaseFunction):  # also called Tablet
    def __call__(self, x):
        """Class of **Discus** (also called **Tablet**) test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return discus(x)


def cigar_discus(x):
    """**Cigar-Discus** test function.

       .. note:: Its dimensionality should `> 1`.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x = np.square(squeeze_and_check(x, True))
    if x.size == 2:
        y = x[0] + (10.0 ** 4) * np.sum(x) + (10.0 ** 6) * x[-1]
    else:
        y = x[0] + (10.0 ** 4) * np.sum(x[1:-1]) + (10.0 ** 6) * x[-1]
    return y


class CigarDiscus(BaseFunction):
    def __call__(self, x):
        """Class of **Cigar-Discus** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return cigar_discus(x)


def ellipsoid(x):
    """**Ellipsoid** test function.

       .. note:: Its dimensionality should `> 1`.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x = np.square(squeeze_and_check(x, True))
    y = np.dot(np.power(10.0, 6.0 * np.linspace(0.0, 1.0, x.size)), x)
    return y


class Ellipsoid(BaseFunction):
    def __call__(self, x):
        """Class of **Ellipsoid** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return ellipsoid(x)


def different_powers(x):
    """**Different-Powers** test function.

       .. note:: Its dimensionality should `> 1`.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x = np.abs(squeeze_and_check(x, True))
    y = np.sum(np.power(x, 2.0 + 4.0 * np.linspace(0.0, 1.0, x.size)))
    return y


class DifferentPowers(BaseFunction):
    def __call__(self, x):
        """Class of **Different-Powers** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return different_powers(x)


def schwefel221(x):
    """**Schwefel221** test function.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    y = np.max(np.abs(squeeze_and_check(x)))
    return y


class Schwefel221(BaseFunction):
    def __call__(self, x):
        """Class of **Schwefel221** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return schwefel221(x)


def step(x):
    """**Step** test function.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    y = np.sum(np.square(np.floor(squeeze_and_check(x) + 0.5)))
    return y


class Step(BaseFunction):
    def __call__(self, x):
        """Class of **Step** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return step(x)


def schwefel222(x):
    """**Schwefel222** test function.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x = np.abs(squeeze_and_check(x))
    y = np.sum(x) + np.prod(x)
    return y


class Schwefel222(BaseFunction):
    def __call__(self, x):
        """Class of **Schwefel222** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return schwefel222(x)


def rosenbrock(x):
    """**Rosenbrock** test function.

       .. note:: Its dimensionality should `> 1`.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x = squeeze_and_check(x, True)
    y = 100.0 * np.sum(np.square(x[1:] - np.square(x[:-1]))) + np.sum(np.square(x[:-1] - 1.0))
    return y


class Rosenbrock(BaseFunction):
    def __call__(self, x):
        """Class of **Rosenbrock** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return rosenbrock(x)


def schwefel12(x):
    """**Schwefel12** test function.

       .. note:: Its dimensionality should `> 1`.

    Parameters
    ----------
    x : ndarray
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x = squeeze_and_check(x, True)
    x = [np.sum(x[:i + 1]) for i in range(x.size)]
    y = np.sum(np.square(x))
    return y


class Schwefel12(BaseFunction):
    def __call__(self, x):
        """Class of **Schwefel12** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return schwefel12(x)


def exponential(x):
    """**Exponential** test function.

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
    y = -np.exp(-0.5 * np.sum(np.square(x)))
    return y


class Exponential(BaseFunction):
    def __call__(self, x):
        """Class of **Exponential** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return exponential(x)


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

       .. note:: It's LaTeX formulation is `$10 n + \sum_{i = 1}^{n} (x_i^2 - 10 \cos(2 \pi x_i))$`.

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
    y = 10.0 * x.size + np.sum(np.square(x) - 10.0 * np.cos(2.0 * np.pi * x))
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
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x, w = squeeze_and_check(x), np.power(10.0, np.linspace(0.0, 1.0, len(x)))
    x *= w
    y = 10.0 * len(x) + np.sum(np.square(x) - 10.0 * np.cos(2.0 * np.pi * x))
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
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x, y = squeeze_and_check(x), 0.0
    for i in range(x.size):
        y -= np.sin(x[i]) * np.power(np.sin((i + 1) * np.square(x[i]) / np.pi), 20)
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
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x = np.sqrt(np.sum(np.square(squeeze_and_check(x))))
    return 1.0 - np.cos(2.0 * np.pi * x) + 0.1 * x


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
        input vector.

    Returns
    -------
    y : float
        scalar fitness.
    """
    x, y = squeeze_and_check(x), 0.0
    for i in range(x.size - 1):
        xx = np.power(x[i], 2) + np.power(x[i + 1], 2)
        y += np.power(xx, 0.25) * (np.power(np.sin(50.0 * np.power(xx, 0.1)), 2) + 1.0)
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


# all of the following functions are only for visualization purpose
def cosine(x):
    """**Cosine** test function.

    Parameters
    ----------
    x: ndarray
       input vector.

    Returns
    -------
    y: float
       scalar fitness.
    """
    y = 10.0 * x[0] ** 2 * (1.0 + 0.75 * math.cos(70.0 * x[0]) / 12.0) + math.cos(100.0 * x[0]) ** 2 / 24.0 + \
        2.0 * x[1] ** 2 * (1.0 + 0.75 * math.cos(70.0 * x[1]) / 12.0) + math.cos(100.0 * x[1]) ** 2 / 24.0 + \
        4.0 * x[0] * x[1]
    return y


def dennis_woods(x):
    """**Dennis-Woods** test function.

    Parameters
    ----------
    x: ndarray
       input vector.

    Returns
    -------
    y: float
       scalar fitness.

    References
    ----------
    Dennis, J. E., Daniel J. Woods, 1987.
    Optimization on microcomputers: The Nelder-Mead simplex algorithm.
    New computing environments: microcomputers in large-scale computing, 11, p. 6-122.
    """
    c_1 = np.array([1.0, -1.0])
    y = 0.5 * max(np.linalg.norm(x - c_1) ** 2, np.linalg.norm(x + c_1) ** 2)
    return y
