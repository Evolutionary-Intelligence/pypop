import numpy as np

from benchmarks import base_functions
from benchmarks.base_functions import BaseFunction
from benchmarks.shifted_functions import _load_shift_vector
from benchmarks.rotated_functions import _load_rotation_matrix


# helper functions
def load_shift_and_rotation(func, x, shift_vector=None, rotation_matrix=None):
    shift_vector = _load_shift_vector(func, x, shift_vector)
    rotation_matrix = _load_rotation_matrix(func, x, rotation_matrix)
    return shift_vector, rotation_matrix


def sphere(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(sphere, x, shift_vector, rotation_matrix)
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
    shift_vector, rotation_matrix = load_shift_and_rotation(cigar, x, shift_vector, rotation_matrix)
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
    shift_vector, rotation_matrix = load_shift_and_rotation(discus, x, shift_vector, rotation_matrix)
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
    shift_vector, rotation_matrix = load_shift_and_rotation(cigar_discus, x, shift_vector, rotation_matrix)
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
    shift_vector, rotation_matrix = load_shift_and_rotation(ellipsoid, x, shift_vector, rotation_matrix)
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
    shift_vector, rotation_matrix = load_shift_and_rotation(different_powers, x, shift_vector, rotation_matrix)
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
    shift_vector, rotation_matrix = load_shift_and_rotation(schwefel221, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.schwefel221(x)
    return y


class Schwefel221(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel221'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return schwefel221(x, shift_vector, rotation_matrix)


def step(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(step, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.step(x)
    return y


class Step(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'step'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return step(x, shift_vector, rotation_matrix)


def schwefel222(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(schwefel222, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.schwefel222(x)
    return y


class Schwefel222(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel222'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return schwefel222(x, shift_vector, rotation_matrix)


def rosenbrock(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(rosenbrock, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.rosenbrock(x)
    return y


class Rosenbrock(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'rosenbrock'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return rosenbrock(x, shift_vector, rotation_matrix)


def schwefel12(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(schwefel12, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.schwefel12(x)
    return y


class Schwefel12(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel12'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return schwefel12(x, shift_vector, rotation_matrix)


def exponential(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(exponential, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.exponential(x)
    return y


class Exponential(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'exponential'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return exponential(x, shift_vector, rotation_matrix)


def griewank(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(griewank, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.griewank(x)
    return y


class Griewank(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'griewank'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return griewank(x, shift_vector, rotation_matrix)


def ackley(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(ackley, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.ackley(x)
    return y


class Ackley(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'ackley'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return ackley(x, shift_vector, rotation_matrix)


def rastrigin(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(rastrigin, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.rastrigin(x)
    return y


class Rastrigin(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'rastrigin'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return rastrigin(x, shift_vector, rotation_matrix)


def levy_montalvo(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(levy_montalvo, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.levy_montalvo(x)
    return y


class LevyMontalvo(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'levy_montalvo'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return levy_montalvo(x, shift_vector, rotation_matrix)


def michalewicz(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(michalewicz, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.michalewicz(x)
    return y


class Michalewicz(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'michalewicz'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return michalewicz(x, shift_vector, rotation_matrix)
