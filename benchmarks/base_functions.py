import numpy as np


# helper function
def _squeeze_and_check(x, size_gt_1=False):
    """Squeeze the input `x` into 1-d `numpy.ndarray`.
        And check whether its number of dimensions == 1. If not, raise a TypeError.
        Optionally, check whether its size > 1. If not, raise a TypeError.
    """
    x = np.squeeze(x)
    if (x.ndim == 0) and (x.size == 1):
        x = np.array([x])
    if x.ndim != 1:
        raise TypeError(f"The number of dimensions should == 1 (not {x.ndim}) after numpy.squeeze(x).")
    if size_gt_1 and not (x.size > 1):
        raise TypeError(f"The size should > 1 (not {x.size}) after numpy.squeeze(x).")
    if x.size == 0:
        raise TypeError(f"the size should != 0.")
    return x


def sphere(x):
    y = np.sum(np.power(_squeeze_and_check(x), 2))
    return y


def cigar(x):
    x = np.power(_squeeze_and_check(x, True), 2)
    y = x[0] + (10 ** 6) * np.sum(x[1:])
    return y


def discus(x):  # also called tablet
    x = np.power(_squeeze_and_check(x, True), 2)
    y = (10 ** 6) * x[0] + np.sum(x[1:])
    return y
