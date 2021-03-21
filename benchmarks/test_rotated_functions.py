import unittest
import os
import numpy as np

from benchmarks.base_functions import sphere as base_sphere
from benchmarks.rotated_functions import sphere
from benchmarks.rotated_functions import _generate_rotation_matrix, _load_rotation_matrix
from benchmarks.test_base_functions import Sample


# helper function
def _check_rotation_matrix(x, error=1e-6):
    x = np.mat(x)
    if x.shape[0] != x.shape[1]:
        raise TypeError('rotation matrix should be a square matrix.')
    a = np.abs(np.matmul(x, x.transpose()) - np.eye(x.shape[0])) < error
    b = np.abs(np.matmul(x.transpose(), x) - np.eye(x.shape[0])) < error
    c = np.abs(np.sum(np.power(x, 2), 0) - 1) < error
    d = np.abs(np.sum(np.power(x, 2), 1) - 1) < error
    e = np.linalg.matrix_rank(x) == x.shape[0]
    return np.all(a) and np.all(b) and np.all(c) and np.all(d) and e


class RotatedSample(Sample):
    """Test correctness of rotated function via sampling (test cases).
    """
    def __init__(self, ndim=None):
        Sample.__init__(self, ndim)

    def compare_rotated_func_values(self, func, ndim, y_true, atol=1e-08, rotation_matrix=None):
        """Compare true (expected) function values with these returned (computed) by rotated function.

        :param func: rotated function, a function object.
        :param ndim: number of dimensions, an `int` scalar ranged in [1, 7].
        :param y_true: a 1-d `ndarray`, where each element is the true function value of the corresponding test case.
        :param atol: absolute tolerance parameter, a `float` scalar.
        :param rotation_matrix: rotation matrix, array_like of floats.
        :return: `True` if all function values computed on test cases match `y_true`; otherwise, `False`.
        """
        x = self.make_test_cases(ndim)
        y = np.empty((x.shape[0],))
        for i in range(x.shape[0]):
            rotation_matrix = _load_rotation_matrix(func, x[i], rotation_matrix)
            y[i] = func(np.dot(np.linalg.inv(rotation_matrix), x[i]))
        return np.allclose(y, y_true, atol=atol)


class Test(unittest.TestCase):
    def test_generate_rotation_matrix(self):
        func, ndim, seed = 'sphere', 100, 0
        x = _generate_rotation_matrix(func, ndim, seed)
        self.assertTrue(_check_rotation_matrix(x))
        data_folder = 'pypop_benchmarks_input_data'
        data_path = os.path.join(data_folder, 'rotation_matrix_' + func + '_dim_' + str(ndim) + '.txt')
        x = np.loadtxt(data_path)
        self.assertTrue(_check_rotation_matrix(x))

    def test_load_rotation_matrix(self):
        func, ndim, seed = base_sphere, 2, 1
        _generate_rotation_matrix(func, ndim, seed)
        rotation_matrix = [[7.227690004350708630e-01, 6.910896989610599839e-01],
                           [6.910896989610598729e-01, -7.227690004350709740e-01]]
        self.assertTrue(np.allclose(_load_rotation_matrix(func, np.ones(ndim,)), rotation_matrix))
        rotation_matrix = np.eye(ndim)
        self.assertTrue(np.allclose(_load_rotation_matrix(func, np.ones(ndim,), rotation_matrix), rotation_matrix))

    def test_sphere(self):
        for ndim in range(1, 8):
            _generate_rotation_matrix(sphere, ndim, 0)
        rotated_sample = RotatedSample()
        x1 = [4, 1, 0, 1, 4]
        self.assertTrue(rotated_sample.compare_rotated_func_values(sphere, 1, x1))
        x2 = [8, 2, 0, 2, 8]
        self.assertTrue(rotated_sample.compare_rotated_func_values(sphere, 2, x2))
        x3 = [12, 3, 0, 3, 12]
        self.assertTrue(rotated_sample.compare_rotated_func_values(sphere, 3, x3))
        x4 = [0, 4, 4, 4, 30, 30, 30]
        self.assertTrue(rotated_sample.compare_rotated_func_values(sphere, 4, x4))
        x5 = [0, 5, 5, 5, 55, 55, 55]
        self.assertTrue(rotated_sample.compare_rotated_func_values(sphere, 5, x5))
        x6 = [0, 6, 6, 6, 91, 91, 91]
        self.assertTrue(rotated_sample.compare_rotated_func_values(sphere, 6, x6))
        x7 = [0, 7, 7, 7, 140, 140, 140, 91]
        self.assertTrue(rotated_sample.compare_rotated_func_values(sphere, 7, x7))


if __name__ == '__main__':
    unittest.main()
