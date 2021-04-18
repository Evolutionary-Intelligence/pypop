import numpy as np


class TestCases(object):
    """Test correctness of benchmark functions via sampling (test cases).
    """
    def __init__(self, ndim=None):
        self.ndim = ndim

    def make_test_cases(self, ndim=None):
        """Make multiple test cases for a specific dimension ranged in [1, 7].

        Note that the number of test cases may be different for different dimensions.

        :param ndim: number of dimensions, an `int` scalar ranged in [1, 7].
        :return: a 2-d `ndarray` of dtype `np.float64`, where each row is a test case.
        """
        if ndim is not None:
            self.ndim = ndim

        if ndim == 1:
            x = [[-2],
                 [-1],
                 [0],
                 [1],
                 [2]]
        elif ndim == 2:
            x = [[-2, 2],
                 [-1, 1],
                 [0, 0],
                 [1, 1],
                 [2, 2]]
        elif ndim == 3:
            x = [[-2, 2, 2],
                 [-1, 1, 1],
                 [0, 0, 0],
                 [1, 1, -1],
                 [2, 2, -2]]
        elif ndim == 4:
            x = [[0, 0, 0, 0],
                 [1, 1, 1, 1],
                 [-1, -1, -1, -1],
                 [1, -1, 1, -1],
                 [1, 2, 3, 4],
                 [1, -2, 3, -4],
                 [-4, 3, 2, -1]]
        elif ndim == 5:
            x = [[0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1],
                 [-1, -1, -1, -1, -1],
                 [1, -1, 1, -1, 1],
                 [1, 2, 3, 4, 5],
                 [1, -2, 3, -4, 5],
                 [-5, 4, 3, 2, -1]]
        elif ndim == 6:
            x = [[0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [-1, -1, -1, -1, -1, -1],
                 [1, -1, 1, -1, 1, 1],
                 [1, 2, 3, 4, 5, 6],
                 [1, -2, 3, -4, 5, -6],
                 [-6, 5, 4, 3, 2, -1]]
        elif ndim == 7:
            x = [[0, 0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1, 1],
                 [-1, -1, -1, -1, -1, -1, -1],
                 [1, -1, 1, -1, 1, 1, -1],
                 [1, 2, 3, 4, 5, 6, 7],
                 [1, -2, 3, -4, 5, -6, 7],
                 [-7, 6, 5, 4, 3, 2, -1],
                 [0, 1, 2, 3, 4, 5, 6]]
        else:
            raise TypeError('The number of dimensions should >=1 and <= 7.')
        return np.array(x, dtype=np.float64)

    def compare(self, func, ndim, y_true, atol=1e-08):
        """Compare true (expected) function values with these returned (computed) by benchmark function.

        :param func: benchmark function, a function object.
        :param ndim: number of dimensions, an `int` scalar ranged in [1, 7].
        :param y_true: a 1-d `ndarray`, where each element is the true function value of the corresponding test case.
        :param atol: absolute tolerance parameter, a `float` scalar.
        :return: `True` if all function values computed on test cases match `y_true`; otherwise, `False`.
        """
        x = self.make_test_cases(ndim)
        y = np.empty((x.shape[0],))
        for i in range(x.shape[0]):
            y[i] = func(x[i])
        return np.allclose(y, y_true, atol=atol)

    def check_origin(self, func, n_samples=7):
        """Check the origin point at which the function value is zero via random sampling (test cases).

        :param func: rotated function, a function object.
        :param n_samples: number of samples, an `int` scalar.
        :return: `True` if all function values computed on test cases are zeros; otherwise, `False`.
        """
        ndims = np.random.default_rng().integers(2, 1000, size=(n_samples,))
        self.ndim = ndims
        is_zero = True
        for d in ndims:
            if np.abs(func(np.zeros((d,)))) > 1e-9:
                is_zero = False
                break
        return is_zero
