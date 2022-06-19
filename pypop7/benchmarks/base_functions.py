import numpy as np


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
    def __init__(self):
        pass


def sphere(x):
    y = np.sum(np.power(squeeze_and_check(x), 2))
    return y


class Sphere(BaseFunction):
    def __call__(self, x):
        return sphere(x)


def cigar(x):
    x = np.power(squeeze_and_check(x, True), 2)
    y = x[0] + (10 ** 6) * np.sum(x[1:])
    return y


class Cigar(BaseFunction):
    def __call__(self, x):
        return cigar(x)


def discus(x):  # also called tablet
    x = np.power(squeeze_and_check(x, True), 2)
    y = (10 ** 6) * x[0] + np.sum(x[1:])
    return y


class Discus(BaseFunction):  # also called Tablet
    def __call__(self, x):
        return discus(x)


def cigar_discus(x):
    x = np.power(squeeze_and_check(x, True), 2)
    if x.size == 2:
        y = x[0] + (10 ** 4) * np.sum(x) + (10 ** 6) * x[-1]
    else:
        y = x[0] + (10 ** 4) * np.sum(x[1:-1]) + (10 ** 6) * x[-1]
    return y


class CigarDiscus(BaseFunction):
    def __call__(self, x):
        return cigar_discus(x)


def ellipsoid(x):
    x = np.power(squeeze_and_check(x, True), 2)
    y = np.dot(np.power(10, 6 * np.linspace(0, 1, x.size)), x)
    return y


class Ellipsoid(BaseFunction):
    def __call__(self, x):
        return ellipsoid(x)


def different_powers(x):
    x = np.abs(squeeze_and_check(x, True))
    y = np.sum(np.power(x, 2 + 4 * np.linspace(0, 1, x.size)))
    return y


class DifferentPowers(BaseFunction):
    def __call__(self, x):
        return different_powers(x)


def schwefel221(x):
    y = np.max(np.abs(squeeze_and_check(x)))
    return y


class Schwefel221(BaseFunction):
    def __call__(self, x):
        return schwefel221(x)


def step(x):
    y = np.sum(np.power(np.floor(squeeze_and_check(x) + 0.5), 2))
    return y


class Step(BaseFunction):
    def __call__(self, x):
        return step(x)


def schwefel222(x):
    x = np.abs(squeeze_and_check(x))
    y = np.sum(x) + np.prod(x)
    return y


class Schwefel222(BaseFunction):
    def __call__(self, x):
        return schwefel222(x)


def rosenbrock(x):
    x = squeeze_and_check(x, True)
    y = 100 * np.sum(np.power(x[1:] - np.power(x[:-1], 2), 2)) + np.sum(np.power(x[:-1] - 1, 2))
    return y


class Rosenbrock(BaseFunction):
    def __call__(self, x):
        return rosenbrock(x)


def schwefel12(x):
    x = squeeze_and_check(x, True)
    x = [np.sum(x[:i + 1]) for i in range(x.size)]
    y = np.sum(np.power(x, 2))
    return y


class Schwefel12(BaseFunction):
    def __call__(self, x):
        return schwefel12(x)


def exponential(x):
    x = squeeze_and_check(x)
    y = -np.exp(-0.5 * np.sum(np.power(x, 2)))
    return y


class Exponential(BaseFunction):
    def __call__(self, x):
        return exponential(x)


def griewank(x):
    x = squeeze_and_check(x)
    y = np.sum(np.power(x, 2)) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1)))) + 1
    return y


class Griewank(BaseFunction):
    def __call__(self, x):
        return griewank(x)


def bohachevsky(x):
    x, y = squeeze_and_check(x), 0
    for i in range(x.size - 1):
        y += np.power(x[i], 2) + 2 * np.power(x[i + 1], 2) -\
             0.3 * np.cos(3 * np.pi * x[i]) - 0.4 * np.cos(4 * np.pi * x[i + 1]) + 0.7
    return y


class Bohachevsky(BaseFunction):
    def __call__(self, x):
        return bohachevsky(x)


def ackley(x):
    x = squeeze_and_check(x)
    y = -20 * np.exp(-0.2 * np.sqrt(np.sum(np.power(x, 2)) / x.size)) -\
        np.exp(np.sum(np.cos(2 * np.pi * x)) / x.size) +\
        20 + np.exp(1)
    return y


class Ackley(BaseFunction):
    def __call__(self, x):
        return ackley(x)


def rastrigin(x):
    x = squeeze_and_check(x)
    y = 10 * x.size + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x))
    return y


class Rastrigin(BaseFunction):
    def __call__(self, x):
        return rastrigin(x)


def scaled_rastrigin(x):
    x, w = squeeze_and_check(x), np.power(10, np.linspace(0, 1, x.size))
    x *= w
    y = 10 * x.size + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x))
    return y


class ScaledRastrigin(BaseFunction):
    def __call__(self, x):
        return scaled_rastrigin(x)


def skew_rastrigin(x):
    x = squeeze_and_check(x)
    for i in range(x.size):
        if x[i] > 0:
            x[i] *= 10
    y = rastrigin(x)
    return y


class SkewRastrigin(BaseFunction):
    def __call__(self, x):
        return skew_rastrigin(x)


def levy_montalvo(x):
    x, y = 1 + (1 / 4) * (squeeze_and_check(x) + 1), 0
    for i in range(x.size - 1):
        y += np.power(x[i] - 1, 2) * (1 + 10 * np.power(np.sin(np.pi * x[i + 1]), 2))
    y += 10 * np.power(np.sin(np.pi * x[0]), 2) + np.power(x[-1] - 1, 2)
    return (np.pi / x.size) * y


class LevyMontalvo(BaseFunction):
    def __call__(self, x):
        return levy_montalvo(x)


def michalewicz(x):
    x, y = squeeze_and_check(x), 0
    for i in range(x.size):
        y -= np.sin(x[i]) * np.power(np.sin((i + 1) * np.power(x[i], 2) / np.pi), 20)
    return y


class Michalewicz(BaseFunction):
    def __call__(self, x):
        return michalewicz(x)


def salomon(x):
    x = np.sqrt(np.sum(np.power(squeeze_and_check(x), 2)))
    y = 1 - np.cos(2 * np.pi * x) + 0.1 * x
    return y


class Salomon(BaseFunction):
    def __call__(self, x):
        return salomon(x)


def shubert(x):
    x, y = squeeze_and_check(x), 1
    for i in range(x.size):
        yy = 0
        for j in range(1, 6):
            yy += j * np.cos((j + 1) * x[i] + j)
        y *= yy
    return y


class Shubert(BaseFunction):
    def __call__(self, x):
        return shubert(x)


def schaffer(x):
    x, y = squeeze_and_check(x), 0
    for i in range(x.size - 1):
        xx = np.power(x[i], 2) + np.power(x[i + 1], 2)
        y += np.power(xx, 0.25) * (np.power(np.sin(50 * np.power(xx, 0.1)), 2) + 1)
    return y


class Schaffer(BaseFunction):
    def __call__(self, x):
        return schaffer(x)
