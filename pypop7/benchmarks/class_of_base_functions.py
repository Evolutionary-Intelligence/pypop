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
