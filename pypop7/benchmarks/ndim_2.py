import math

import numpy as np


def cosine(x):
    """**Cosine** test function.

    Parameters
    ----------
    x: ndarray
       An input vector.

    Returns
    -------
    y: float
       A scalar fitness.
    """
    y = 10.0 * x[0] ** 2 * (1.0 + 0.75 * math.cos(70.0 * x[0]) / 12.0) + math.cos(100.0 * x[0]) ** 2 / 24.0 + \
        2.0 * x[1] ** 2 * (1.0 + 0.75 * math.cos(70.0 * x[1]) / 12.0) + math.cos(100.0 * x[1]) ** 2 / 24.0 + \
        4.0 * x[0] * x[1]
    return y


def dennis_woods(x):
    """**Dennis-Woods** test function.

    Parameters
    ----------
    x: ndarray
       An input vector.

    Returns
    -------
    y: float
       A scalar fitness.

    References
    ----------
    Dennis, J. E., Daniel J. Woods, 1987.
    Optimization on microcomputers: The Nelder-Mead simplex algorithm.
    New computing environments: Microcomputers in large-scale computing, 11, p. 6-122.
    """
    c_1 = np.array([1.0, -1.0])
    y = 0.5 * max(np.linalg.norm(x - c_1) ** 2, np.linalg.norm(x + c_1) ** 2)
    return y
