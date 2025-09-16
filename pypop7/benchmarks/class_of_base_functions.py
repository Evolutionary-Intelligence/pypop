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


class Ackley(BaseFunction):
    def __call__(self, x):
        """Class of **Ackley** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return ackley(x)


class Bohachevsky(BaseFunction):
    def __call__(self, x):
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
        return bohachevsky(x)


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


class Exponential(BaseFunction):
    def __call__(self, x):
        """Class of **Exponential** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.exponential(x)


class Griewank(BaseFunction):
    def __call__(self, x):
        """Class of **Griewank** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.griewank(x)


class LevyMontalvo(BaseFunction):
    def __call__(self, x):
        """Class of **Levy-Montalvo** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return bf.levy_montalvo(x)


class Michalewicz(BaseFunction):
    def __call__(self, x):
        """Class of **Michalewicz** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return bf.michalewicz(x)


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


class Rosenbrock(BaseFunction):
    def __call__(self, x):
        """Class of **Rosenbrock** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.rosenbrock(x)


class Salomon(BaseFunction):
    def __call__(self, x):
        """Class of **Salomon** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.salomon(x)


class ScaledRastrigin(BaseFunction):
    def __call__(self, x):
        """Class of **Scaled-Rastrigin** test function.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return scaled_rastrigin(x)


class Schaffer(BaseFunction):
    def __call__(self, x):
        """Class of **Schaffer** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.schaffer(x)


class Schwefel12(BaseFunction):
    def __call__(self, x):
        """Class of **Schwefel12** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.schwefel12(x)


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


class Shubert(BaseFunction):
    def __call__(self, x):
        """Class of **Shubert** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.shubert(x)


class SkewRastrigin(BaseFunction):
    def __call__(self, x):
        """Class of **Skew-Rastrigin** test function.

        Parameters
        ----------
        x : ndarray
            An input vector.

        Returns
        -------
        y : float
            A scalar fitness.
        """
        return bf.skew_rastrigin(x)


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
