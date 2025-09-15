from pypop7.benchmarks import base_functions as bf


# helper class
class BaseFunction(object):
    """Class for all base functions.

    Parameters
    ----------
    x : ndarray
        An input vector.

    Returns
    -------
    y : float
        A scalar fitness.
    """
    def __init__(self):
        pass

    def __call__(self, x):
        pass


class Cigar(BaseFunction):
    def __call__(self, x):
        """Class of **Cigar** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.cigar(x)


class CigarDiscus(BaseFunction):
    def __call__(self, x):
        """Class of **Cigar-Discus** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.cigar_discus(x)


class DifferentPowers(BaseFunction):
    def __call__(self, x):
        """Class of **Different-Powers** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.different_powers(x)


class Discus(BaseFunction):
    def __call__(self, x):
        """Class of **Discus** (also called **Tablet**) test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.discus(x)


class Ellipsoid(BaseFunction):
    def __call__(self, x):
        """Class of **Ellipsoid** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.ellipsoid(x)


class Rastrigin(BaseFunction):
    def __call__(self, x):
        """Class of **Rastrigin** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.rastrigin(x)


class Schwefel221(BaseFunction):
    def __call__(self, x):
        """Class of **Schwefel221** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.schwefel221(x)


class Schwefel222(BaseFunction):
    def __call__(self, x):
        """Class of **Schwefel222** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.schwefel222(x)


class Sphere(BaseFunction):
    def __call__(self, x):
        """Class of **Sphere** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.sphere(x)


class Step(BaseFunction):
    def __call__(self, x):
        """Class of **Step** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.step(x)
