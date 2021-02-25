import unittest
import numpy as np

from base_functions import sphere as base_sphere
from shifted_functions import sphere, cigar, discus, cigar_discus, ellipsoid, different_powers, schwefel221,\
    rosenbrock, Sphere
from shifted_functions import _generate_shift_vector, _load_shift_vector
from test_base_functions import Sample


class ShiftedSample(Sample):
    """Test correctness of shifted function via sampling (test cases).
    """
    def __init__(self, ndim=None):
        Sample.__init__(self, ndim)

    def compare_shifted_func_values(self, func, ndim, y_true, atol=1e-08, shift_vector=None):
        """Compare true (expected) function values with these returned (computed) by shifted function.

        :param func: shifted function, a function object.
        :param ndim: number of dimensions, an `int` scalar ranged in [1, 7].
        :param y_true: a 1-d `ndarray`, where each element is the true function value of the corresponding test case.
        :param atol: absolute tolerance parameter, a `float` scalar.
        :param shift_vector: shift vector, array_like of floats.
        :return: `True` if all function values computed on test cases match `y_true`; otherwise, `False`.
        """
        x = self.make_test_cases(ndim)
        y = np.empty((x.shape[0],))
        for i in range(x.shape[0]):
            y[i] = func(x[i] + _load_shift_vector(func, x[i], shift_vector))
        return np.allclose(y, y_true, atol=atol)


class TestShiftedFunctions(unittest.TestCase):
    def test_generate_shift_vector(self):
        shift_vector = _generate_shift_vector('sphere', 2, -5, 10, 0)
        self.assertEqual(shift_vector.size, 2)
        self.assertTrue(np.all(shift_vector >= -5))
        self.assertTrue(np.all(shift_vector < 10))
        self.assertTrue(np.allclose(shift_vector, [4.554425309821814594e+00, -9.531992935419451030e-01]))

        shift_vector = _generate_shift_vector(base_sphere, 2, [-1, -2], [1, 2], 0)
        self.assertEqual(shift_vector.size, 2)
        self.assertTrue(np.all(shift_vector >= [-1, -2]))
        self.assertTrue(np.all(shift_vector < [1, 2]))
        self.assertTrue(np.allclose(shift_vector, [2.739233746429086125e-01, -9.208531449445187533e-01]))

        shift_vector = _generate_shift_vector(base_sphere, 1, -100, 100, 7)
        self.assertEqual(shift_vector.size, 1)
        self.assertTrue(np.all(shift_vector >= -100))
        self.assertTrue(np.all(shift_vector < 100))
        self.assertTrue(np.allclose(shift_vector, 2.501909332093339344e+01))

    def test_load_shift_vector(self):
        func = base_sphere
        _generate_shift_vector(func, 2, [-1, -2], [1, 2], 0)
        shift_vector = _load_shift_vector(func, [0, 0])
        self.assertTrue(np.allclose(shift_vector, [2.739233746429086125e-01, -9.208531449445187533e-01]))

        _generate_shift_vector(func, 3, -100, 100, 7)
        shift_vector = _load_shift_vector(func, [0, 0, 0])
        self.assertTrue(np.allclose(shift_vector,
                                    [2.501909332093339344e+01, 7.944276019391509180e+01, 5.513713804903869686e+01]))

        shift_vector = _load_shift_vector(func, 0, 77)
        self.assertTrue(np.allclose(shift_vector, 77))

    def test_sphere(self):
        for ndim in range(1, 8):
            _generate_shift_vector(sphere, ndim, -100, 100, seed=0)
        shifted_sample = ShiftedSample()
        x1 = [4, 1, 0, 1, 4]
        self.assertTrue(shifted_sample.compare_shifted_func_values(sphere, 1, x1))
        x2 = [8, 2, 0, 2, 8]
        self.assertTrue(shifted_sample.compare_shifted_func_values(sphere, 2, x2))
        x3 = [12, 3, 0, 3, 12]
        self.assertTrue(shifted_sample.compare_shifted_func_values(sphere, 3, x3))
        x4 = [0, 4, 4, 4, 30, 30, 30]
        self.assertTrue(shifted_sample.compare_shifted_func_values(sphere, 4, x4))
        x5 = [0, 5, 5, 5, 55, 55, 55]
        self.assertTrue(shifted_sample.compare_shifted_func_values(sphere, 5, x5))
        x6 = [0, 6, 6, 6, 91, 91, 91]
        self.assertTrue(shifted_sample.compare_shifted_func_values(sphere, 6, x6))
        x7 = [0, 7, 7, 7, 140, 140, 140, 91]
        self.assertTrue(shifted_sample.compare_shifted_func_values(sphere, 7, x7))

    def test_Sphere(self):
        sphere_object = Sphere()
        for ndim in range(1, 8):
            _generate_shift_vector(sphere, ndim, -100, 100, seed=0)
        shifted_sample = ShiftedSample()
        x1 = [4, 1, 0, 1, 4]
        self.assertTrue(shifted_sample.compare_shifted_func_values(sphere_object, 1, x1))
        x2 = [8, 2, 0, 2, 8]
        self.assertTrue(shifted_sample.compare_shifted_func_values(sphere_object, 2, x2))
        x3 = [12, 3, 0, 3, 12]
        self.assertTrue(shifted_sample.compare_shifted_func_values(sphere_object, 3, x3))
        x4 = [0, 4, 4, 4, 30, 30, 30]
        self.assertTrue(shifted_sample.compare_shifted_func_values(sphere_object, 4, x4))
        x5 = [0, 5, 5, 5, 55, 55, 55]
        self.assertTrue(shifted_sample.compare_shifted_func_values(sphere_object, 5, x5))
        x6 = [0, 6, 6, 6, 91, 91, 91]
        self.assertTrue(shifted_sample.compare_shifted_func_values(sphere_object, 6, x6))
        x7 = [0, 7, 7, 7, 140, 140, 140, 91]
        self.assertTrue(shifted_sample.compare_shifted_func_values(sphere_object, 7, x7))

    def test_cigar(self):
        for ndim in range(1, 8):
            _generate_shift_vector(cigar, ndim, -100, 100, seed=1)
        shifted_sample = ShiftedSample()
        x2 = [4000004, 1000001, 0, 1000001, 4000004]
        self.assertTrue(shifted_sample.compare_shifted_func_values(cigar, 2, x2))
        x3 = [8000004, 2000001, 0, 2000001, 8000004]
        self.assertTrue(shifted_sample.compare_shifted_func_values(cigar, 3, x3))
        x4 = [0, 3000001, 3000001, 3000001, 29000001, 29000001, 14000016]
        self.assertTrue(shifted_sample.compare_shifted_func_values(cigar, 4, x4))
        x5 = [0, 4000001, 4000001, 4000001, 54000001, 54000001, 30000025]
        self.assertTrue(shifted_sample.compare_shifted_func_values(cigar, 5, x5))
        x6 = [0, 5000001, 5000001, 5000001, 90000001, 90000001, 55000036]
        self.assertTrue(shifted_sample.compare_shifted_func_values(cigar, 6, x6))
        x7 = [0, 6000001, 6000001, 6000001, 139000001, 139000001, 91000049, 91000000]
        self.assertTrue(shifted_sample.compare_shifted_func_values(cigar, 7, x7))
        with self.assertRaisesRegex(TypeError, "The size should > 1+"):
            shifted_sample.compare_shifted_func_values(cigar, 1, np.empty((5,)))

    def test_discus(self):
        for ndim in range(1, 8):
            _generate_shift_vector(discus, ndim, -100, 100, seed=2)
        shifted_sample = ShiftedSample()
        x2 = [4000004, 1000001, 0, 1000001, 4000004]
        self.assertTrue(shifted_sample.compare_shifted_func_values(discus, 2, x2))
        x3 = [4000008, 1000002, 0, 1000002, 4000008]
        self.assertTrue(shifted_sample.compare_shifted_func_values(discus, 3, x3))
        x4 = [0, 1000003, 1000003, 1000003, 1000029, 1000029, 16000014]
        self.assertTrue(shifted_sample.compare_shifted_func_values(discus, 4, x4))
        x5 = [0, 1000004, 1000004, 1000004, 1000054, 1000054, 25000030]
        self.assertTrue(shifted_sample.compare_shifted_func_values(discus, 5, x5))
        x6 = [0, 1000005, 1000005, 1000005, 1000090, 1000090, 36000055]
        self.assertTrue(shifted_sample.compare_shifted_func_values(discus, 6, x6))
        x7 = [0, 1000006, 1000006, 1000006, 1000139, 1000139, 49000091, 91]
        self.assertTrue(shifted_sample.compare_shifted_func_values(discus, 7, x7))
        with self.assertRaisesRegex(TypeError, "The size should > 1+"):
            shifted_sample.compare_shifted_func_values(discus, 1, np.empty((5,)))

    def test_cigar_discus(self):
        for ndim in range(1, 8):
            _generate_shift_vector(cigar_discus, ndim, -100, 100, seed=3)
        shifted_sample = ShiftedSample()
        x2 = [4080004, 1020001, 0, 1020001, 4080004]
        self.assertTrue(shifted_sample.compare_shifted_func_values(cigar_discus, 2, x2))
        x3 = [4040004, 1010001, 0, 1010001, 4040004]
        self.assertTrue(shifted_sample.compare_shifted_func_values(cigar_discus, 3, x3))
        x4 = [0, 1020001, 1020001, 1020001, 16130001, 16130001, 1130016]
        self.assertTrue(shifted_sample.compare_shifted_func_values(cigar_discus, 4, x4))
        x5 = [0, 1030001, 1030001, 1030001, 25290001, 25290001, 1290025]
        self.assertTrue(shifted_sample.compare_shifted_func_values(cigar_discus, 5, x5))
        x6 = [0, 1040001, 1040001, 1040001, 36540001, 36540001, 1540036]
        self.assertTrue(shifted_sample.compare_shifted_func_values(cigar_discus, 6, x6))
        x7 = [0, 1050001, 1050001, 1050001, 49900001, 49900001, 1900049, 36550000]
        self.assertTrue(shifted_sample.compare_shifted_func_values(cigar_discus, 7, x7))
        with self.assertRaisesRegex(TypeError, "The size should > 1+"):
            shifted_sample.compare_shifted_func_values(cigar_discus, 1, np.empty((5,)))

    def test_ellipsoid(self):
        for ndim in range(1, 8):
            _generate_shift_vector(ellipsoid, ndim, -100, 100, seed=4)
        shifted_sample = ShiftedSample()
        x2 = [4000004, 1000001, 0, 1000001, 4000004]
        self.assertTrue(shifted_sample.compare_shifted_func_values(ellipsoid, 2, x2))
        x3 = [4004004, 1001001, 0, 1001001, 4004004]
        self.assertTrue(shifted_sample.compare_shifted_func_values(ellipsoid, 3, x3))
        x4 = [0, 1010101, 1010101, 1010101, 16090401, 16090401, 1040916]
        self.assertTrue(shifted_sample.compare_shifted_func_values(ellipsoid, 4, x4))
        x5 = [0, 1032655, 1032655, 1032655, 25515092, 25515092, 1136022]
        self.assertTrue(shifted_sample.compare_shifted_func_values(ellipsoid, 5, x5))
        x6 = [0, 1067345, 1067345, 1067345, 37643416, 37643416, 1292664]
        self.assertTrue(shifted_sample.compare_shifted_func_values(ellipsoid, 6, x6))
        x7 = [0, 1111111, 1111111, 1111111, 52866941, 52866941, 1508909, 38669410]
        self.assertTrue(shifted_sample.compare_shifted_func_values(ellipsoid, 7, x7))
        with self.assertRaisesRegex(TypeError, "The size should > 1+"):
            shifted_sample.compare_shifted_func_values(ellipsoid, 1, np.empty((5,)))

    def test_different_powers(self):
        for ndim in range(1, 8):
            _generate_shift_vector(different_powers, ndim, -100, 100, seed=5)
        shifted_sample = ShiftedSample()
        x2 = [68, 2, 0, 2, 68]
        self.assertTrue(shifted_sample.compare_shifted_func_values(different_powers, 2, x2))
        x3 = [84, 3, 0, 3, 84]
        self.assertTrue(shifted_sample.compare_shifted_func_values(different_powers, 3, x3))
        x4 = [0, 4, 4, 4, 4275.6, 4275.6, 81.3]
        self.assertTrue(shifted_sample.compare_shifted_func_values(different_powers, 4, x4, 0.1))
        x5 = [0, 5, 5, 5, 16739, 16739, 203]
        self.assertTrue(shifted_sample.compare_shifted_func_values(different_powers, 5, x5))
        x6 = [0, 6, 6, 6, 51473.5, 51473.5, 437.1]
        self.assertTrue(shifted_sample.compare_shifted_func_values(different_powers, 6, x6, 0.1))
        x7 = [0, 7, 7, 7, 133908.7, 133908.7, 847.4, 52736.8]
        self.assertTrue(shifted_sample.compare_shifted_func_values(different_powers, 7, x7, 0.1))
        with self.assertRaisesRegex(TypeError, "The size should > 1+"):
            shifted_sample.compare_shifted_func_values(different_powers, 1, np.empty((5,)))

    def test_schwefel221(self):
        for ndim in range(1, 8):
            _generate_shift_vector(schwefel221, ndim, -100, 100, seed=6)
        shifted_sample = ShiftedSample()
        x1 = [2, 1, 0, 1, 2]
        self.assertTrue(shifted_sample.compare_shifted_func_values(schwefel221, 1, x1))
        x2 = [2, 1, 0, 1, 2]
        self.assertTrue(shifted_sample.compare_shifted_func_values(schwefel221, 2, x2))
        x3 = [2, 1, 0, 1, 2]
        self.assertTrue(shifted_sample.compare_shifted_func_values(schwefel221, 3, x3))
        x4 = [0, 1, 1, 1, 4, 4, 4]
        self.assertTrue(shifted_sample.compare_shifted_func_values(schwefel221, 4, x4))
        x5 = [0, 1, 1, 1, 5, 5, 5]
        self.assertTrue(shifted_sample.compare_shifted_func_values(schwefel221, 5, x5))
        x6 = [0, 1, 1, 1, 6, 6, 6]
        self.assertTrue(shifted_sample.compare_shifted_func_values(schwefel221, 6, x6))
        x7 = [0, 1, 1, 1, 7, 7, 7, 6]
        self.assertTrue(shifted_sample.compare_shifted_func_values(schwefel221, 7, x7))

    def test_rosenbrock(self):
        for ndim in range(1, 8):
            _generate_shift_vector(rosenbrock, ndim, -100, 100, seed=7)
        shifted_sample = ShiftedSample()
        x2 = [409, 4, 1, 0, 401]
        self.assertTrue(shifted_sample.compare_shifted_func_values(rosenbrock, 2, x2))
        x3 = [810, 4, 2, 400, 4002]
        self.assertTrue(shifted_sample.compare_shifted_func_values(rosenbrock, 3, x3))
        x4 = [3, 0, 1212, 804, 2705, 17913, 24330]
        self.assertTrue(shifted_sample.compare_shifted_func_values(rosenbrock, 4, x4))
        x5 = [4, 0, 1616, 808, 14814, 30038, 68450]
        self.assertTrue(shifted_sample.compare_shifted_func_values(rosenbrock, 5, x5))
        x6 = [5, 0, 2020, 808, 50930, 126154, 164579]
        self.assertTrue(shifted_sample.compare_shifted_func_values(rosenbrock, 6, x6))
        x7 = [6, 0, 2424, 1208, 135055, 210303, 349519, 51031]
        self.assertTrue(shifted_sample.compare_shifted_func_values(rosenbrock, 7, x7))
        with self.assertRaisesRegex(TypeError, "The size should > 1+"):
            shifted_sample.compare_shifted_func_values(rosenbrock, 1, np.empty((5,)))


if __name__ == '__main__':
    unittest.main()
