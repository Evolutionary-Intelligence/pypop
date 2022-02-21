import os
import numpy as np

from benchmarks import base_functions
from benchmarks.base_functions import _squeeze_and_check, BaseFunction


# helper functions
def generate_rotation_matrix(func, ndim, seed):
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


class Sphere(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'sphere'

    def __call__(self, x, rotation_matrix=None):
        return sphere(x, rotation_matrix)


def cigar(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(cigar, x, rotation_matrix)
    y = base_functions.cigar(np.dot(rotation_matrix, x))
    return y


class Cigar(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'cigar'

    def __call__(self, x, rotation_matrix=None):
        return cigar(x, rotation_matrix)


def discus(x, rotation_matrix=None):  # also called tablet
    rotation_matrix = _load_rotation_matrix(discus, x, rotation_matrix)
    y = base_functions.discus(np.dot(rotation_matrix, x))
    return y


class Discus(BaseFunction):  # also called Tablet
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'discus'

    def __call__(self, x, rotation_matrix=None):
        return discus(x, rotation_matrix)


def cigar_discus(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(cigar_discus, x, rotation_matrix)
    y = base_functions.cigar_discus(np.dot(rotation_matrix, x))
    return y


class CigarDiscus(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'cigar_discus'

    def __call__(self, x, rotation_matrix=None):
        return cigar_discus(x, rotation_matrix)


def ellipsoid(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(ellipsoid, x, rotation_matrix)
    y = base_functions.ellipsoid(np.dot(rotation_matrix, x))
    return y


class Ellipsoid(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'ellipsoid'

    def __call__(self, x, rotation_matrix=None):
        return ellipsoid(x, rotation_matrix)


def different_powers(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(different_powers, x, rotation_matrix)
    y = base_functions.different_powers(np.dot(rotation_matrix, x))
    return y


class DifferentPowers(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'different_powers'

    def __call__(self, x, rotation_matrix=None):
        return different_powers(x, rotation_matrix)


def schwefel221(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(schwefel221, x, rotation_matrix)
    y = base_functions.schwefel221(np.dot(rotation_matrix, x))
    return y


class Schwefel221(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel221'

    def __call__(self, x, rotation_matrix=None):
        return schwefel221(x, rotation_matrix)


def step(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(step, x, rotation_matrix)
    y = base_functions.step(np.dot(rotation_matrix, x))
    return y


class Step(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'step'

    def __call__(self, x, rotation_matrix=None):
        return step(x, rotation_matrix)


def schwefel222(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(schwefel222, x, rotation_matrix)
    y = base_functions.schwefel222(np.dot(rotation_matrix, x))
    return y


class Schwefel222(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel222'

    def __call__(self, x, rotation_matrix=None):
        return schwefel222(x, rotation_matrix)


def rosenbrock(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(rosenbrock, x, rotation_matrix)
    y = base_functions.rosenbrock(np.dot(rotation_matrix, x))
    return y


class Rosenbrock(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'rosenbrock'

    def __call__(self, x, rotation_matrix=None):
        return rosenbrock(x, rotation_matrix)


def schwefel12(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(schwefel12, x, rotation_matrix)
    y = base_functions.schwefel12(np.dot(rotation_matrix, x))
    return y


class Schwefel12(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel12'

    def __call__(self, x, rotation_matrix=None):
        return schwefel12(x, rotation_matrix)


def exponential(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(exponential, x, rotation_matrix)
    y = base_functions.exponential(np.dot(rotation_matrix, x))
    return y


class Exponential(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'exponential'

    def __call__(self, x, rotation_matrix=None):
        return exponential(x, rotation_matrix)


def griewank(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(griewank, x, rotation_matrix)
    y = base_functions.griewank(np.dot(rotation_matrix, x))
    return y


class Griewank(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'griewank'

    def __call__(self, x, rotation_matrix=None):
        return griewank(x, rotation_matrix)


def bohachevsky(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(bohachevsky, x, rotation_matrix)
    y = base_functions.bohachevsky(np.dot(rotation_matrix, x))
    return y


class Bohachevsky(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'bohachevsky'

    def __call__(self, x, rotation_matrix=None):
        return bohachevsky(x, rotation_matrix)


def ackley(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(ackley, x, rotation_matrix)
    y = base_functions.ackley(np.dot(rotation_matrix, x))
    return y


class Ackley(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'ackley'

    def __call__(self, x, rotation_matrix=None):
        return ackley(x, rotation_matrix)


def rastrigin(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(rastrigin, x, rotation_matrix)
    y = base_functions.rastrigin(np.dot(rotation_matrix, x))
    return y


class Rastrigin(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'rastrigin'

    def __call__(self, x, rotation_matrix=None):
        return rastrigin(x, rotation_matrix)


def scaled_rastrigin(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(scaled_rastrigin, x, rotation_matrix)
    y = base_functions.scaled_rastrigin(np.dot(rotation_matrix, x))
    return y


class ScaledRastrigin(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'scaled_rastrigin'

    def __call__(self, x, rotation_matrix=None):
        return scaled_rastrigin(x, rotation_matrix)


def skew_rastrigin(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(skew_rastrigin, x, rotation_matrix)
    y = base_functions.skew_rastrigin(np.dot(rotation_matrix, x))
    return y


class SkewRastrigin(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'skew_rastrigin'

    def __call__(self, x, rotation_matrix=None):
        return skew_rastrigin(x, rotation_matrix)


def levy_montalvo(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(levy_montalvo, x, rotation_matrix)
    y = base_functions.levy_montalvo(np.dot(rotation_matrix, x))
    return y


class LevyMontalvo(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'levy_montalvo'

    def __call__(self, x, rotation_matrix=None):
        return levy_montalvo(x, rotation_matrix)


def michalewicz(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(michalewicz, x, rotation_matrix)
    y = base_functions.michalewicz(np.dot(rotation_matrix, x))
    return y


class Michalewicz(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'michalewicz'

    def __call__(self, x, rotation_matrix=None):
        return michalewicz(x, rotation_matrix)


def salomon(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(salomon, x, rotation_matrix)
    y = base_functions.salomon(np.dot(rotation_matrix, x))
    return y


class Salomon(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'salomon'

    def __call__(self, x, rotation_matrix=None):
        return salomon(x, rotation_matrix)


def shubert(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(shubert, x, rotation_matrix)
    y = base_functions.shubert(np.dot(rotation_matrix, x))
    return y


class Shubert(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'shubert'

    def __call__(self, x, rotation_matrix=None):
        return shubert(x, rotation_matrix)


def schaffer(x, rotation_matrix=None):
    rotation_matrix = _load_rotation_matrix(schaffer, x, rotation_matrix)
    y = base_functions.schaffer(np.dot(rotation_matrix, x))
    return y


class Schaffer(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schaffer'

    def __call__(self, x, rotation_matrix=None):
        return schaffer(x, rotation_matrix)
