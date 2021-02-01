import unittest
import numpy as np

from base_functions import _squeeze_and_check, sphere, cigar, discus, cigar_discus, ellipsoid, different_powers,\
    schwefel221


class Sample(object):
    """Test correctness of base function via sampling (test cases).
    """
    def __init__(self, ndim=None):
        self.ndim = ndim

    def make_test_cases(self, ndim=None):
        """Make multiple test cases for a specific dimension.

        Note that the number of test cases may be different for different dimensions.

        :param ndim: number of dimensions, an `int` scalar ranged in [1, 7].
        :return: a 2-d `ndarray` of `dtype` `np.float64`, where each row is a test case.
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
            raise TypeError("The number of dimensions should >=1 and <= 7.")
        return np.array(x, dtype=np.float64)

    def compare_func_values(self, func, ndim, y_true, atol=1e-08):
        """Compare true (expected) function values with these returned (computed) by base function.

        :param func: base function, a function object.
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
        self.assertTrue(sample.compare_func_values(sphere, 1, x1))
        x2 = [8, 2, 0, 2, 8]
        self.assertTrue(sample.compare_func_values(sphere, 2, x2))
        x3 = [12, 3, 0, 3, 12]
        self.assertTrue(sample.compare_func_values(sphere, 3, x3))
        x4 = [0, 4, 4, 4, 30, 30, 30]
        self.assertTrue(sample.compare_func_values(sphere, 4, x4))
        x5 = [0, 5, 5, 5, 55, 55, 55]
        self.assertTrue(sample.compare_func_values(sphere, 5, x5))
        x6 = [0, 6, 6, 6, 91, 91, 91]
        self.assertTrue(sample.compare_func_values(sphere, 6, x6))
        x7 = [0, 7, 7, 7, 140, 140, 140, 91]
        self.assertTrue(sample.compare_func_values(sphere, 7, x7))

    def test_cigar(self):
        sample = Sample()
        x2 = [4000004, 1000001, 0, 1000001, 4000004]
        self.assertTrue(sample.compare_func_values(cigar, 2, x2))
        x3 = [8000004, 2000001, 0, 2000001, 8000004]
        self.assertTrue(sample.compare_func_values(cigar, 3, x3))
        x4 = [0, 3000001, 3000001, 3000001, 29000001, 29000001, 14000016]
        self.assertTrue(sample.compare_func_values(cigar, 4, x4))
        x5 = [0, 4000001, 4000001, 4000001, 54000001, 54000001, 30000025]
        self.assertTrue(sample.compare_func_values(cigar, 5, x5))
        x6 = [0, 5000001, 5000001, 5000001, 90000001, 90000001, 55000036]
        self.assertTrue(sample.compare_func_values(cigar, 6, x6))
        x7 = [0, 6000001, 6000001, 6000001, 139000001, 139000001, 91000049, 91000000]
        self.assertTrue(sample.compare_func_values(cigar, 7, x7))
        with self.assertRaisesRegex(TypeError, "The size should > 1+"):
            sample.compare_func_values(cigar, 1, np.empty((5,)))

    def test_discus(self):
        sample = Sample()
        x2 = [4000004, 1000001, 0, 1000001, 4000004]
        self.assertTrue(sample.compare_func_values(discus, 2, x2))
        x3 = [4000008, 1000002, 0, 1000002, 4000008]
        self.assertTrue(sample.compare_func_values(discus, 3, x3))
        x4 = [0, 1000003, 1000003, 1000003, 1000029, 1000029, 16000014]
        self.assertTrue(sample.compare_func_values(discus, 4, x4))
        x5 = [0, 1000004, 1000004, 1000004, 1000054, 1000054, 25000030]
        self.assertTrue(sample.compare_func_values(discus, 5, x5))
        x6 = [0, 1000005, 1000005, 1000005, 1000090, 1000090, 36000055]
        self.assertTrue(sample.compare_func_values(discus, 6, x6))
        x7 = [0, 1000006, 1000006, 1000006, 1000139, 1000139, 49000091, 91]
        self.assertTrue(sample.compare_func_values(discus, 7, x7))
        with self.assertRaisesRegex(TypeError, "The size should > 1+"):
            sample.compare_func_values(discus, 1, np.empty((5,)))

    def test_cigar_discus(self):
        sample = Sample()
        x2 = [4080004, 1020001, 0, 1020001, 4080004]
        self.assertTrue(sample.compare_func_values(cigar_discus, 2, x2))
        x3 = [4040004, 1010001, 0, 1010001, 4040004]
        self.assertTrue(sample.compare_func_values(cigar_discus, 3, x3))
        x4 = [0, 1020001, 1020001, 1020001, 16130001, 16130001, 1130016]
        self.assertTrue(sample.compare_func_values(cigar_discus, 4, x4))
        x5 = [0, 1030001, 1030001, 1030001, 25290001, 25290001, 1290025]
        self.assertTrue(sample.compare_func_values(cigar_discus, 5, x5))
        x6 = [0, 1040001, 1040001, 1040001, 36540001, 36540001, 1540036]
        self.assertTrue(sample.compare_func_values(cigar_discus, 6, x6))
        x7 = [0, 1050001, 1050001, 1050001, 49900001, 49900001, 1900049, 36550000]
        self.assertTrue(sample.compare_func_values(cigar_discus, 7, x7))
        with self.assertRaisesRegex(TypeError, "The size should > 1+"):
            sample.compare_func_values(cigar_discus, 1, np.empty((5,)))

    def test_ellipsoid(self):
        sample = Sample()
        x2 = [4000004, 1000001, 0, 1000001, 4000004]
        self.assertTrue(sample.compare_func_values(ellipsoid, 2, x2))
        x3 = [4004004, 1001001, 0, 1001001, 4004004]
        self.assertTrue(sample.compare_func_values(ellipsoid, 3, x3))
        x4 = [0, 1010101, 1010101, 1010101, 16090401, 16090401, 1040916]
        self.assertTrue(sample.compare_func_values(ellipsoid, 4, x4))
        x5 = [0, 1032655, 1032655, 1032655, 25515092, 25515092, 1136022]
        self.assertTrue(sample.compare_func_values(ellipsoid, 5, x5))
        x6 = [0, 1067345, 1067345, 1067345, 37643416, 37643416, 1292664]
        self.assertTrue(sample.compare_func_values(ellipsoid, 6, x6))
        x7 = [0, 1111111, 1111111, 1111111, 52866941, 52866941, 1508909, 38669410]
        self.assertTrue(sample.compare_func_values(ellipsoid, 7, x7))
        with self.assertRaisesRegex(TypeError, "The size should > 1+"):
            sample.compare_func_values(ellipsoid, 1, np.empty((5,)))

    def test_different_powers(self):
        sample = Sample()
        x2 = [68, 2, 0, 2, 68]
        self.assertTrue(sample.compare_func_values(different_powers, 2, x2))
        x3 = [84, 3, 0, 3, 84]
        self.assertTrue(sample.compare_func_values(different_powers, 3, x3))
        x4 = [0, 4, 4, 4, 4275.6, 4275.6, 81.3]
        self.assertTrue(sample.compare_func_values(different_powers, 4, x4, 0.1))
        x5 = [0, 5, 5, 5, 16739, 16739, 203]
        self.assertTrue(sample.compare_func_values(different_powers, 5, x5))
        x6 = [0, 6, 6, 6, 51473.5, 51473.5, 437.1]
        self.assertTrue(sample.compare_func_values(different_powers, 6, x6, 0.1))
        x7 = [0, 7, 7, 7, 133908.7, 133908.7, 847.4, 52736.8]
        self.assertTrue(sample.compare_func_values(different_powers, 7, x7, 0.1))
        with self.assertRaisesRegex(TypeError, "The size should > 1+"):
            sample.compare_func_values(different_powers, 1, np.empty((5,)))

    def test_schwefel221(self):
        sample = Sample()
        x1 = [2, 1, 0, 1, 2]
        self.assertTrue(sample.compare_func_values(schwefel221, 1, x1))
        x2 = [2, 1, 0, 1, 2]
        self.assertTrue(sample.compare_func_values(schwefel221, 2, x2))
        x3 = [2, 1, 0, 1, 2]
        self.assertTrue(sample.compare_func_values(schwefel221, 3, x3))
        x4 = [0, 1, 1, 1, 4, 4, 4]
        self.assertTrue(sample.compare_func_values(schwefel221, 4, x4))
        x5 = [0, 1, 1, 1, 5, 5, 5]
        self.assertTrue(sample.compare_func_values(schwefel221, 5, x5))
        x6 = [0, 1, 1, 1, 6, 6, 6]
        self.assertTrue(sample.compare_func_values(schwefel221, 6, x6))
        x7 = [0, 1, 1, 1, 7, 7, 7, 6]
        self.assertTrue(sample.compare_func_values(schwefel221, 7, x7))


if __name__ == '__main__':
    unittest.main()
