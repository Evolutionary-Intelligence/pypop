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

       .. note:: It's LaTeX formulation is `$\sum_{i=1}^{n}x_i^2$`.

    Parameters
    ----------
    x : input vector, `ndarray`.

    Returns
    -------
    y : scalar, `float`.
    """
    y = np.sum(np.square(squeeze_and_check(x)))
    return y


class Sphere(BaseFunction):
    def __call__(self, x):
        return sphere(x)


def cigar(x):
    """**Cigar** test function.

       .. note:: It's LaTeX formulation is ``. Its dimensionality should `> 1`.

    """
    x = np.square(squeeze_and_check(x, True))
    y = x[0] + (10.0 ** 6) * np.sum(x[1:])
    return y


class Cigar(BaseFunction):
    def __call__(self, x):
        return cigar(x)


def discus(x):  # also called tablet
    """**Discus** test function.

       .. note:: It's LaTeX formulation is ``. Its dimensionality should `> 1`.

    """
    x = np.square(squeeze_and_check(x, True))
    y = (10.0 ** 6) * x[0] + np.sum(x[1:])
    return y


class Discus(BaseFunction):  # also called Tablet
    def __call__(self, x):
        return discus(x)


def cigar_discus(x):
    """**Cigar-Discus** test function.

       .. note:: It's LaTeX formulation is ``. Its dimensionality should `> 1`.

    """
    x = np.square(squeeze_and_check(x, True))
    if x.size == 2:
        y = x[0] + (10.0 ** 4) * np.sum(x) + (10.0 ** 6) * x[-1]
    else:
        y = x[0] + (10.0 ** 4) * np.sum(x[1:-1]) + (10.0 ** 6) * x[-1]
    return y


class CigarDiscus(BaseFunction):
    def __call__(self, x):
        return cigar_discus(x)


def ellipsoid(x):
    """**Ellipsoid** test function.

       .. note:: It's LaTeX formulation is ``. Its dimensionality should `> 1`.

    """
    x = np.square(squeeze_and_check(x, True))
    y = np.dot(np.power(10.0, 6.0 * np.linspace(0.0, 1.0, x.size)), x)
    return y


class Ellipsoid(BaseFunction):
    def __call__(self, x):
        return ellipsoid(x)


def different_powers(x):
    """**Different-Powers** test function.

       .. note:: It's LaTeX formulation is ``. Its dimensionality should `> 1`.

    """
    x = np.abs(squeeze_and_check(x, True))
    y = np.sum(np.power(x, 2.0 + 4.0 * np.linspace(0.0, 1.0, x.size)))
    return y


class DifferentPowers(BaseFunction):
    def __call__(self, x):
        return different_powers(x)


def schwefel221(x):
    """**Schwefel221** test function.

       .. note:: It's LaTeX formulation is ``.

    """
    y = np.max(np.abs(squeeze_and_check(x)))
    return y


class Schwefel221(BaseFunction):
    def __call__(self, x):
        return schwefel221(x)


def step(x):
    """**Step** test function.

       .. note:: It's LaTeX formulation is ``.

    """
    y = np.sum(np.square(np.floor(squeeze_and_check(x) + 0.5)))
    return y


class Step(BaseFunction):
    def __call__(self, x):
        return step(x)


def schwefel222(x):
    """**Schwefel222** test function.

       .. note:: It's LaTeX formulation is ``.

    """
    x = np.abs(squeeze_and_check(x))
    y = np.sum(x) + np.prod(x)
    return y


class Schwefel222(BaseFunction):
    def __call__(self, x):
        return schwefel222(x)


def rosenbrock(x):
    """**Rosenbrock** test function.

       .. note:: It's LaTeX formulation is ``. Its dimensionality should `> 1`.

    """
    x = squeeze_and_check(x, True)
    y = 100.0 * np.sum(np.square(x[1:] - np.square(x[:-1]))) + np.sum(np.square(x[:-1] - 1.0))
    return y


class Rosenbrock(BaseFunction):
    def __call__(self, x):
        return rosenbrock(x)


def schwefel12(x):
    """**Schwefel12** test function.

       .. note:: It's LaTeX formulation is ``. Its dimensionality should `> 1`.

    """
    x = squeeze_and_check(x, True)
    x = [np.sum(x[:i + 1]) for i in range(x.size)]
    y = np.sum(np.square(x))
    return y


class Schwefel12(BaseFunction):
    def __call__(self, x):
        return schwefel12(x)


def exponential(x):
    """**Exponential** test function.

       .. note:: It's LaTeX formulation is ``.

    """
    x = squeeze_and_check(x)
    y = -np.exp(-0.5 * np.sum(np.square(x)))
    return y


class Exponential(BaseFunction):
    def __call__(self, x):
        return exponential(x)


def griewank(x):
    """**Griewank** test function.

       .. note:: It's LaTeX formulation is ``.

    """
    x = squeeze_and_check(x)
    y = np.sum(np.square(x)) / 4000.0 - np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1)))) + 1.0
    return y


class Griewank(BaseFunction):
    def __call__(self, x):
        return griewank(x)


def bohachevsky(x):
    """**Bohachevsky** test function.

       .. note:: It's LaTeX formulation is ``.

    """
    x, y = squeeze_and_check(x), 0.0
    for i in range(x.size - 1):
        y += np.square(x[i]) + 2.0 * np.square(x[i + 1]) - 0.3 * np.cos(3.0 * np.pi * x[i]) - \
             0.4 * np.cos(4.0 * np.pi * x[i + 1]) + 0.7
    return y


class Bohachevsky(BaseFunction):
    def __call__(self, x):
        return bohachevsky(x)


def ackley(x):
    """**Ackley** test function.

       .. note:: It's LaTeX formulation is ``.

    """
    x = squeeze_and_check(x)
    y = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(np.square(x)) / x.size)) - \
        np.exp(np.sum(np.cos(2.0 * np.pi * x)) / x.size) + 20.0 + np.exp(1)
    return y


class Ackley(BaseFunction):
    def __call__(self, x):
        return ackley(x)


def rastrigin(x):
    """**Rastrigin** test function.

       .. note:: It's LaTeX formulation is ``.

    """
    x = squeeze_and_check(x)
    y = 10.0 * x.size + np.sum(np.square(x) - 10.0 * np.cos(2.0 * np.pi * x))
    return y


class Rastrigin(BaseFunction):
    def __call__(self, x):
        return rastrigin(x)


def scaled_rastrigin(x):
    """**Scaled-Rastrigin** test function.

       .. note:: It's LaTeX formulation is ``.

    """
    x, w = squeeze_and_check(x), np.power(10.0, np.linspace(0.0, 1.0, x.size))
    x *= w
    y = 10.0 * x.size + np.sum(np.square(x) - 10.0 * np.cos(2.0 * np.pi * x))
    return y


class ScaledRastrigin(BaseFunction):
    def __call__(self, x):
        return scaled_rastrigin(x)


def skew_rastrigin(x):
    """**Skew-Rastrigin** test function.

       .. note:: It's LaTeX formulation is ``.

    """
    x = squeeze_and_check(x)
    for i in range(x.size):
        if x[i] > 0.0:
            x[i] *= 10.0
    y = rastrigin(x)
    return y


class SkewRastrigin(BaseFunction):
    def __call__(self, x):
        return skew_rastrigin(x)


def levy_montalvo(x):
    """**Levy-Montalvo** test function.

       .. note:: It's LaTeX formulation is ``.

    """
    x, y = 1.0 + 0.25 * (squeeze_and_check(x) + 1.0), 0.0
    for i in range(x.size - 1):
        y += np.square(x[i] - 1.0) * (1.0 + 10.0 * np.square(np.sin(np.pi * x[i + 1])))
    y += 10.0 * np.square(np.sin(np.pi * x[0])) + np.square(x[-1] - 1.0)
    return (np.pi / x.size) * y


class LevyMontalvo(BaseFunction):
    def __call__(self, x):
        return levy_montalvo(x)


def michalewicz(x):
    """**Michalewicz** test function.

       .. note:: It's LaTeX formulation is ``.

    """
    x, y = squeeze_and_check(x), 0.0
    for i in range(x.size):
        y -= np.sin(x[i]) * np.power(np.sin((i + 1) * np.square(x[i]) / np.pi), 20)
    return y


class Michalewicz(BaseFunction):
    def __call__(self, x):
        return michalewicz(x)


def salomon(x):
    """**Salomon** test function.

       .. note:: It's LaTeX formulation is ``.

    """
    x = np.sqrt(np.sum(np.square(squeeze_and_check(x))))
    return 1.0 - np.cos(2.0 * np.pi * x) + 0.1 * x


class Salomon(BaseFunction):
    def __call__(self, x):
        return salomon(x)


def shubert(x):
    """**Shubert** test function.

       .. note:: It's LaTeX formulation is ``.

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
        return shubert(x)


def schaffer(x):
    """**Schaffer** test function.

       .. note:: It's LaTeX formulation is ``.

    Parameters
    ----------
    x : input vector, `ndarray`.

    Returns
    -------
    y : scalar, `float`.
    """
    x, y = squeeze_and_check(x), 0
    for i in range(x.size - 1):
        xx = np.power(x[i], 2) + np.power(x[i + 1], 2)
        y += np.power(xx, 0.25) * (np.power(np.sin(50 * np.power(xx, 0.1)), 2) + 1)
    return y


class Schaffer(BaseFunction):
    def __call__(self, x):
        """

        Parameters
        ----------
        x : input vector, `ndarray`.

        Returns
        -------
        y : scalar, `float`.
        """
        return schaffer(x)
