import numpy as np

from benchmarks import base_functions
from benchmarks.base_functions import BaseFunction
from benchmarks.shifted_functions import _load_shift_vector
from benchmarks.rotated_functions import _load_rotation_matrix


# helper functions
def _load_shift_and_rotation(func, x, shift_vector=None, rotation_matrix=None):
    shift_vector = _load_shift_vector(func, x, shift_vector)
    rotation_matrix = _load_rotation_matrix(func, x, rotation_matrix)
    return shift_vector, rotation_matrix


def sphere(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = _load_shift_and_rotation(sphere, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.sphere(x)
    return y


class Sphere(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'sphere'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return sphere(x, shift_vector, rotation_matrix)


def cigar(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = _load_shift_and_rotation(cigar, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.cigar(x)
    return y


class Cigar(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'cigar'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return cigar(x, shift_vector, rotation_matrix)


def discus(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = _load_shift_and_rotation(discus, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.discus(x)
    return y


class Discus(BaseFunction):  # also called Tablet
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'discus'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return discus(x, shift_vector, rotation_matrix)


def cigar_discus(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = _load_shift_and_rotation(cigar_discus, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.cigar_discus(x)
    return y


class CigarDiscus(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'cigar_discus'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return cigar_discus(x, shift_vector, rotation_matrix)


def ellipsoid(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = _load_shift_and_rotation(ellipsoid, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.ellipsoid(x)
    return y


class Ellipsoid(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'ellipsoid'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return ellipsoid(x, shift_vector, rotation_matrix)


def different_powers(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = _load_shift_and_rotation(different_powers, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.different_powers(x)
    return y


class DifferentPowers(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'different_powers'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return different_powers(x, shift_vector, rotation_matrix)


def schwefel221(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = _load_shift_and_rotation(schwefel221, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.schwefel221(x)
    return y


def rosenbrock(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = _load_shift_and_rotation(rosenbrock, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.rosenbrock(x)
    return y


def schwefel12(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = _load_shift_and_rotation(schwefel12, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.schwefel12(x)
    return y
