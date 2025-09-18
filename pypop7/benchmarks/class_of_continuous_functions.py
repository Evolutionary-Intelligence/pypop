from pypop7.benchmarks.base_functions import BaseFunction


class Sphere(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'sphere'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return sphere(x, shift_vector, rotation_matrix)


class Cigar(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'cigar'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return cigar(x, shift_vector, rotation_matrix)


class Discus(BaseFunction):  # also called Tablet
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'discus'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return discus(x, shift_vector, rotation_matrix)


class CigarDiscus(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'cigar_discus'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return cigar_discus(x, shift_vector, rotation_matrix)


class Ellipsoid(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'ellipsoid'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return ellipsoid(x, shift_vector, rotation_matrix)


class DifferentPowers(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'different_powers'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return different_powers(x, shift_vector, rotation_matrix)


class Schwefel221(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel221'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return schwefel221(x, shift_vector, rotation_matrix)


class Step(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'step'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return step(x, shift_vector, rotation_matrix)


class Schwefel222(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel222'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return schwefel222(x, shift_vector, rotation_matrix)


class Rosenbrock(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'rosenbrock'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return rosenbrock(x, shift_vector, rotation_matrix)


class Schwefel12(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel12'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return schwefel12(x, shift_vector, rotation_matrix)


class Exponential(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'exponential'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return exponential(x, shift_vector, rotation_matrix)


class Griewank(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'griewank'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return griewank(x, shift_vector, rotation_matrix)


class Bohachevsky(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'bohachevsky'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return bohachevsky(x, shift_vector, rotation_matrix)


class Ackley(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'ackley'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return ackley(x, shift_vector, rotation_matrix)


class Rastrigin(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'rastrigin'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return rastrigin(x, shift_vector, rotation_matrix)


class ScaledRastrigin(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'scaled_rastrigin'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return scaled_rastrigin(x, shift_vector, rotation_matrix)


class SkewRastrigin(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'skew_rastrigin'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return skew_rastrigin(x, shift_vector, rotation_matrix)


class LevyMontalvo(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'levy_montalvo'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return levy_montalvo(x, shift_vector, rotation_matrix)


class Michalewicz(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'michalewicz'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return michalewicz(x, shift_vector, rotation_matrix)


class Salomon(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'salomon'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return salomon(x, shift_vector, rotation_matrix)


class Shubert(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'shubert'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return shubert(x, shift_vector, rotation_matrix)


class Schaffer(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schaffer'

    def __call__(self, x, shift_vector=None, rotation_matrix=None):
        return schaffer(x, shift_vector, rotation_matrix)
