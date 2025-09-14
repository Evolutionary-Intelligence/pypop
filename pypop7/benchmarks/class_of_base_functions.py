from pypop7.benchmarks import base_functions as bf


# helper class
class BaseFunction(object):
    """Class for all base functions.
    """
    def __init__(self):
        pass


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
