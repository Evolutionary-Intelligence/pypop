import math


def cosine(x):
    """**Cosine** test function.

    Parameters
    ----------
    x: ndarray
       input vector.

    Returns
    -------
    y: float
       scalar fitness.
    """
    y = 10.0 * x[0] ** 2 * (1.0 + 0.75 * math.cos(70.0 * x[0]) / 12.0) + math.cos(100.0 * x[0]) ** 2 / 24.0 + \
        2.0 * x[1] ** 2 * (1.0 + 0.75 * math.cos(70.0 * x[1]) / 12.0) + math.cos(100.0 * x[1]) ** 2 / 24.0 + \
        4.0 * x[0] * x[1]
    return y
