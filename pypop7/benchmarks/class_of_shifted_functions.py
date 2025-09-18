class Sphere(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'sphere'

    def __call__(self, x, shift_vector=None):
        return sphere(x, shift_vector)


class Cigar(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'cigar'

    def __call__(self, x, shift_vector=None):
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
        return cigar(x, shift_vector)


class Discus(BaseFunction):  # also called Tablet
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'discus'

    def __call__(self, x, shift_vector=None):
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
        return discus(x, shift_vector)


class CigarDiscus(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'cigar_discus'

    def __call__(self, x, shift_vector=None):
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
        return cigar_discus(x, shift_vector)


class Ellipsoid(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'ellipsoid'

    def __call__(self, x, shift_vector=None):
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
        return ellipsoid(x, shift_vector)


class DifferentPowers(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'different_powers'

    def __call__(self, x, shift_vector=None):
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
        return different_powers(x, shift_vector)


class Schwefel221(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel221'

    def __call__(self, x, shift_vector=None):
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
        return schwefel221(x, shift_vector)


class Step(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'step'

    def __call__(self, x, shift_vector=None):
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
        return step(x, shift_vector)


class Schwefel222(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel222'

    def __call__(self, x, shift_vector=None):
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
        return schwefel222(x, shift_vector)


class Rosenbrock(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'rosenbrock'

    def __call__(self, x, shift_vector=None):
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
        return rosenbrock(x, shift_vector)


class Schwefel12(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schwefel12'

    def __call__(self, x, shift_vector=None):
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
        return schwefel12(x, shift_vector)


class Exponential(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'exponential'

    def __call__(self, x, shift_vector=None):
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
        return exponential(x, shift_vector)


class Griewank(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'griewank'

    def __call__(self, x, shift_vector=None):
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
        return griewank(x, shift_vector)


class Bohachevsky(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'bohachevsky'

    def __call__(self, x, shift_vector=None):
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
        return bohachevsky(x, shift_vector)


class Ackley(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'ackley'

    def __call__(self, x, shift_vector=None):
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
        return ackley(x, shift_vector)


class Rastrigin(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'rastrigin'

    def __call__(self, x, shift_vector=None):
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
        return rastrigin(x, shift_vector)


class ScaledRastrigin(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'scaled_rastrigin'

    def __call__(self, x, shift_vector=None):
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
        return scaled_rastrigin(x, shift_vector)


class SkewRastrigin(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'skew_rastrigin'

    def __call__(self, x, shift_vector=None):
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
        return skew_rastrigin(x, shift_vector)


class LevyMontalvo(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'levy_montalvo'

    def __call__(self, x, shift_vector=None):
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
        return levy_montalvo(x, shift_vector)


class Michalewicz(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'michalewicz'

    def __call__(self, x, shift_vector=None):
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
        return michalewicz(x, shift_vector)


class Salomon(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'salomon'

    def __call__(self, x, shift_vector=None):
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
        return salomon(x, shift_vector)


class Shubert(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'shubert'

    def __call__(self, x, shift_vector=None):
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
        return shubert(x, shift_vector)


class Schaffer(BaseFunction):
    def __init__(self):
        BaseFunction.__init__(self)
        self.__name__ = 'schaffer'

    def __call__(self, x, shift_vector=None):
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
        return schaffer(x, shift_vector)
