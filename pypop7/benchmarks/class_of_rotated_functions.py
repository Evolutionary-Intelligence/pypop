class Sphere(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'sphere'

    def __call__(self, x, rotation_matrix=None):
        return sphere(x, rotation_matrix)


class Cigar(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'cigar'

    def __call__(self, x, rotation_matrix=None):
        return cigar(x, rotation_matrix)


class Discus(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'discus'

    def __call__(self, x, rotation_matrix=None):
        return discus(x, rotation_matrix)


class CigarDiscus(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'cigar_discus'

    def __call__(self, x, rotation_matrix=None):
        return cigar_discus(x, rotation_matrix)


class Ellipsoid(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'ellipsoid'

    def __call__(self, x, rotation_matrix=None):
        return ellipsoid(x, rotation_matrix)


class DifferentPowers(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'different_powers'

    def __call__(self, x, rotation_matrix=None):
        return different_powers(x, rotation_matrix)


class Schwefel221(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel221'

    def __call__(self, x, rotation_matrix=None):
        return schwefel221(x, rotation_matrix)


class Step(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'step'

    def __call__(self, x, rotation_matrix=None):
        return step(x, rotation_matrix)


class Schwefel222(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel222'

    def __call__(self, x, rotation_matrix=None):
        return schwefel222(x, rotation_matrix)


class Rosenbrock(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'rosenbrock'

    def __call__(self, x, rotation_matrix=None):
        return rosenbrock(x, rotation_matrix)


class Schwefel12(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel12'

    def __call__(self, x, rotation_matrix=None):
        return schwefel12(x, rotation_matrix)


class Exponential(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'exponential'

    def __call__(self, x, rotation_matrix=None):
        return exponential(x, rotation_matrix)


class Griewank(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'griewank'

    def __call__(self, x, rotation_matrix=None):
        return griewank(x, rotation_matrix)


class Bohachevsky(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'bohachevsky'

    def __call__(self, x, rotation_matrix=None):
        return bohachevsky(x, rotation_matrix)


class Ackley(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'ackley'

    def __call__(self, x, rotation_matrix=None):
        return ackley(x, rotation_matrix)


class Rastrigin(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'rastrigin'

    def __call__(self, x, rotation_matrix=None):
        return rastrigin(x, rotation_matrix)


class ScaledRastrigin(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'scaled_rastrigin'

    def __call__(self, x, rotation_matrix=None):
        return scaled_rastrigin(x, rotation_matrix)


def skew_rastrigin(x, rotation_matrix=None):
    """**Skew-Rastrigin** test function.

       .. note:: It's LaTeX formulation is `$$`.
          If its parameter `rotation_matrix` is `None`, please use function `generate_rotation_matrix()` to
          generate it (stored in *txt* form) in advance.

    Parameters
    ----------
    x               : ndarray
                      input vector.
    rotation_matrix : ndarray
                      a matrix with the same size as `x` in each dimension.

    Returns
    -------
    y : float
        scalar fitness.
    """
    rotation_matrix = load_rotation_matrix(skew_rastrigin, x, rotation_matrix)
    y = base_functions.skew_rastrigin(np.dot(rotation_matrix, x))
    return y


class SkewRastrigin(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'skew_rastrigin'

    def __call__(self, x, rotation_matrix=None):
        return skew_rastrigin(x, rotation_matrix)


class LevyMontalvo(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'levy_montalvo'

    def __call__(self, x, rotation_matrix=None):
        return levy_montalvo(x, rotation_matrix)


class Michalewicz(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'michalewicz'

    def __call__(self, x, rotation_matrix=None):
        return michalewicz(x, rotation_matrix)


class Salomon(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'salomon'

    def __call__(self, x, rotation_matrix=None):
        return salomon(x, rotation_matrix)


class Shubert(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'shubert'

    def __call__(self, x, rotation_matrix=None):
        return shubert(x, rotation_matrix)


class Schaffer(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schaffer'

    def __call__(self, x, rotation_matrix=None):
        return schaffer(x, rotation_matrix)
