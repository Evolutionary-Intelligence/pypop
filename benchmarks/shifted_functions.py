import os
import numpy as np

from benchmarks import base_functions
from benchmarks.base_functions import _squeeze_and_check, BaseFunction


# helper functions
def generate_shift_vector(func, ndim, low, high, seed=None):
    """Generate a random shift vector of dimension `ndim`, sampled uniformly between
        `low` (inclusive) and `high` (exclusive).

    Note that the generated shift vector will be automatically stored in txt form for further use.

    :param func: function name, a `str` or `function` object.
    :param ndim: number of dimensions of the shift vector, an `int` scalar.
    :param low: lower boundary of the shift vector, a `float` scalar or array_like of floats.
    :param high: upper boundary of the shift vector, a `float` scalar or array_like of floats.
    :param seed: seed for random number generator, a `int` scalar.
    :return: shift vector, a `ndim`-d vector sampled uniformly in [`low`, `high`).

    https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.uniform.html
    """
    low, high = _squeeze_and_check(low), _squeeze_and_check(high)
    if hasattr(func, '__call__'):
        func = func.__name__
    data_folder = 'pypop_benchmarks_input_data'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    data_path = os.path.join(data_folder, 'shift_vector_' + func + '_dim_' + str(ndim) + '.txt')
    shift_vector = np.random.default_rng(seed).uniform(low, high, size=ndim)
    np.savetxt(data_path, shift_vector)
    return shift_vector


def _load_shift_vector(func, x, shift_vector=None):
    """Load the shift vector which needs to be generated in advance.
        When `None`, the shift vector should have been generated and stored in txt form in advance.

    :param func: function name, a `function` object.
    :param x: decision vector, array_like of floats.
    :param shift_vector: shift vector, array_like of floats.
    :return: shift vector, a 1-d `ndarray` of `dtype` `np.float64` with the same size as `x`.
    """
    x = _squeeze_and_check(x)
    if shift_vector is None:
        if (not hasattr(func, 'pypop_shift_vector')) or (func.pypop_shift_vector.size != x.size):
            data_folder = 'pypop_benchmarks_input_data'
            data_path = os.path.join(data_folder, 'shift_vector_' + func.__name__ + '_dim_' + str(x.size) + '.txt')
            shift_vector = np.loadtxt(data_path)
            func.pypop_shift_vector = shift_vector
        shift_vector = func.pypop_shift_vector
    shift_vector = _squeeze_and_check(shift_vector)
    if shift_vector.shape != x.shape:
        raise TypeError('shift_vector should have the same shape as x.')
    if shift_vector.size != x.size:
        raise TypeError('shift_vector should have the same size as x.')
    return shift_vector


def sphere(x, shift_vector=None):
    shift_vector = _load_shift_vector(sphere, x, shift_vector)
    y = base_functions.sphere(x - shift_vector)
    return y


class Sphere(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'sphere'

    def __call__(self, x, shift_vector=None):
        return sphere(x, shift_vector)


def cigar(x, shift_vector=None):
    shift_vector = _load_shift_vector(cigar, x, shift_vector)
    y = base_functions.cigar(x - shift_vector)
    return y


class Cigar(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'cigar'

    def __call__(self, x, shift_vector=None):
        return cigar(x, shift_vector)


def discus(x, shift_vector=None):  # also called tablet
    shift_vector = _load_shift_vector(discus, x, shift_vector)
    y = base_functions.discus(x - shift_vector)
    return y


class Discus(BaseFunction):  # also called Tablet
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'discus'

    def __call__(self, x, shift_vector=None):
        return discus(x, shift_vector)


def cigar_discus(x, shift_vector=None):
    shift_vector = _load_shift_vector(cigar_discus, x, shift_vector)
    y = base_functions.cigar_discus(x - shift_vector)
    return y


class CigarDiscus(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'cigar_discus'

    def __call__(self, x, shift_vector=None):
        return cigar_discus(x, shift_vector)


def ellipsoid(x, shift_vector=None):
    shift_vector = _load_shift_vector(ellipsoid, x, shift_vector)
    y = base_functions.ellipsoid(x - shift_vector)
    return y


class Ellipsoid(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'ellipsoid'

    def __call__(self, x, shift_vector=None):
        return ellipsoid(x, shift_vector)


def different_powers(x, shift_vector=None):
    shift_vector = _load_shift_vector(different_powers, x, shift_vector)
    y = base_functions.different_powers(x - shift_vector)
    return y


class DifferentPowers(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'different_powers'

    def __call__(self, x, shift_vector=None):
        return different_powers(x, shift_vector)


def schwefel221(x, shift_vector=None):
    shift_vector = _load_shift_vector(schwefel221, x, shift_vector)
    y = base_functions.schwefel221(x - shift_vector)
    return y


class Schwefel221(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel221'

    def __call__(self, x, shift_vector=None):
        return schwefel221(x, shift_vector)


def step(x, shift_vector=None):
    shift_vector = _load_shift_vector(step, x, shift_vector)
    y = base_functions.step(x - shift_vector)
    return y


class Step(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'step'

    def __call__(self, x, shift_vector=None):
        return step(x, shift_vector)


def schwefel222(x, shift_vector=None):
    shift_vector = _load_shift_vector(schwefel222, x, shift_vector)
    y = base_functions.schwefel222(x - shift_vector)
    return y


class Schwefel222(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel222'

    def __call__(self, x, shift_vector=None):
        return schwefel222(x, shift_vector)


def rosenbrock(x, shift_vector=None):
    shift_vector = _load_shift_vector(rosenbrock, x, shift_vector)
    y = base_functions.rosenbrock(x - shift_vector)
    return y


class Rosenbrock(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'rosenbrock'

    def __call__(self, x, shift_vector=None):
        return rosenbrock(x, shift_vector)


def schwefel12(x, shift_vector=None):
    shift_vector = _load_shift_vector(schwefel12, x, shift_vector)
    y = base_functions.schwefel12(x - shift_vector)
    return y


class Schwefel12(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel12'

    def __call__(self, x, shift_vector=None):
        return schwefel12(x, shift_vector)


def exponential(x, shift_vector=None):
    shift_vector = _load_shift_vector(exponential, x, shift_vector)
    y = base_functions.exponential(x - shift_vector)
    return y


class Exponential(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'exponential'

    def __call__(self, x, shift_vector=None):
        return exponential(x, shift_vector)


def griewank(x, shift_vector=None):
    shift_vector = _load_shift_vector(griewank, x, shift_vector)
    y = base_functions.griewank(x - shift_vector)
    return y


class Griewank(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'griewank'

    def __call__(self, x, shift_vector=None):
        return griewank(x, shift_vector)


def bohachevsky(x, shift_vector=None):
    shift_vector = _load_shift_vector(bohachevsky, x, shift_vector)
    y = base_functions.bohachevsky(x - shift_vector)
    return y


class Bohachevsky(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'bohachevsky'

    def __call__(self, x, shift_vector=None):
        return bohachevsky(x, shift_vector)


def ackley(x, shift_vector=None):
    shift_vector = _load_shift_vector(ackley, x, shift_vector)
    y = base_functions.ackley(x - shift_vector)
    return y


class Ackley(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'ackley'

    def __call__(self, x, shift_vector=None):
        return ackley(x, shift_vector)


def rastrigin(x, shift_vector=None):
    shift_vector = _load_shift_vector(rastrigin, x, shift_vector)
    y = base_functions.rastrigin(x - shift_vector)
    return y


class Rastrigin(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'rastrigin'

    def __call__(self, x, shift_vector=None):
        return rastrigin(x, shift_vector)


def levy_montalvo(x, shift_vector=None):
    shift_vector = _load_shift_vector(levy_montalvo, x, shift_vector)
    y = base_functions.levy_montalvo(x - shift_vector)
    return y


class LevyMontalvo(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'levy_montalvo'

    def __call__(self, x, shift_vector=None):
        return levy_montalvo(x, shift_vector)


def michalewicz(x, shift_vector=None):
    shift_vector = _load_shift_vector(michalewicz, x, shift_vector)
    y = base_functions.michalewicz(x - shift_vector)
    return y


class Michalewicz(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'michalewicz'

    def __call__(self, x, shift_vector=None):
        return michalewicz(x, shift_vector)


def salomon(x, shift_vector=None):
    shift_vector = _load_shift_vector(salomon, x, shift_vector)
    y = base_functions.salomon(x - shift_vector)
    return y


class Salomon(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'salomon'

    def __call__(self, x, shift_vector=None):
        return salomon(x, shift_vector)


def shubert(x, shift_vector=None):
    shift_vector = _load_shift_vector(shubert, x, shift_vector)
    y = base_functions.shubert(x - shift_vector)
    return y


class Shubert(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'shubert'

    def __call__(self, x, shift_vector=None):
        return shubert(x, shift_vector)
