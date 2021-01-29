import unittest
import numpy as np

from base_functions import _squeeze_and_check, sphere


class Sample(object):
    def __init__(self, ndim=None):
        self.ndim = ndim

    def make_cases(self, ndim=None):
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
            raise TypeError("ndim should >=1 and <= 7.")
        return np.array(x, dtype=np.float64)

    def test_cases(self, func, ndim, y_true):
        x = self.make_cases(ndim)
        y = np.empty((x.shape[0],))
        for i in range(x.shape[0]):
            y[i] = func(x[i])
        return np.allclose(y, y_true)


class TestBaseFunctions(unittest.TestCase):
    def test_squeeze_and_check(self):
        self.assertEqual(_squeeze_and_check(0), np.array([0]))
        self.assertEqual(_squeeze_and_check(np.array(0)), np.array([0]))
        x1 = np.array([0.7])
        self.assertEqual(_squeeze_and_check(x1), x1)
        x2 = np.array([0.0, 1.0])
        self.assertTrue(np.allclose(_squeeze_and_check(x2), x2))
        x3 = np.arange(6).reshape(2, 3)
        with self.assertRaisesRegex(TypeError, "The number+"):
            _squeeze_and_check(x3)
        with self.assertRaisesRegex(TypeError, "The size should > 1+"):
            _squeeze_and_check(x1, True)
        with self.assertRaisesRegex(TypeError, "the size should != 0."):
            _squeeze_and_check([])

    def test_sphere(self):
        sample = Sample()
        x1 = [4, 1, 0, 1, 4]
        self.assertTrue(sample.test_cases(sphere, 1, x1))
        x2 = [8, 2, 0, 2, 8]
        self.assertTrue(sample.test_cases(sphere, 2, x2))
        x3 = [12, 3, 0, 3, 12]
        self.assertTrue(sample.test_cases(sphere, 3, x3))
        x4 = [0, 4, 4, 4, 30, 30, 30]
        self.assertTrue(sample.test_cases(sphere, 4, x4))
        x5 = [0, 5, 5, 5, 55, 55, 55]
        self.assertTrue(sample.test_cases(sphere, 5, x5))
        x6 = [0, 6, 6, 6, 91, 91, 91]
        self.assertTrue(sample.test_cases(sphere, 6, x6))
        x7 = [0, 7, 7, 7, 140, 140, 140, 91]
        self.assertTrue(sample.test_cases(sphere, 7, x7))


if __name__ == '__main__':
    unittest.main()
