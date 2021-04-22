import unittest

from benchmarks.base_functions import sphere as base_sphere
from benchmarks.continuous_functions import *
from benchmarks.continuous_functions import _load_shift_and_rotation
from test_cases import *


class Test(unittest.TestCase):
    def test_load_shift_and_rotation(self):
        func, ndim, seed = base_sphere, 2, 1
        generate_shift_vector(func, 2, [-1, -2], [1, 2], 0)
        generate_rotation_matrix(func, ndim, seed)
        shift_vector, rotation_matrix = _load_shift_and_rotation(func, [0, 0])
        self.assertTrue(np.allclose(shift_vector, [2.739233746429086125e-01, -9.208531449445187533e-01]))
        self.assertTrue(np.allclose(rotation_matrix, [[7.227690004350708630e-01, 6.910896989610599839e-01],
                                                      [6.910896989610598729e-01, -7.227690004350709740e-01]]))

    def test_sphere(self):
        sample = TestCases(is_shifted=True, is_rotated=True)
        sphere_object = Sphere()
        for func in [sphere, sphere_object]:
            for ndim in range(1, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_sphere(ndim - 1)))
            self.assertTrue(sample.check_origin(func))

    def test_cigar(self):
        sample = TestCases(is_shifted=True, is_rotated=True)
        cigar_object = Cigar()
        for func in [cigar, cigar_object]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_cigar(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_discus(self):
        sample = TestCases(is_shifted=True, is_rotated=True)
        discus_object = Discus()
        for func in [discus, discus_object]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_discus(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_cigar_discus(self):
        sample = TestCases(is_shifted=True, is_rotated=True)
        cigar_discus_object = CigarDiscus()
        for func in [cigar_discus, cigar_discus_object]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_cigar_discus(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_ellipsoid(self):
        sample = TestCases(is_shifted=True, is_rotated=True)
        ellipsoid_object = Ellipsoid()
        for func in [ellipsoid, ellipsoid_object]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_ellipsoid(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_different_powers(self):
        sample = TestCases(is_shifted=True, is_rotated=True)
        different_powers_object = DifferentPowers()
        for func in [different_powers, different_powers_object]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_different_powers(ndim - 2), atol=0.1))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_schwefel221(self):
        sample = TestCases(is_shifted=True, is_rotated=True)
        schwefel221_object = Schwefel221()
        for func in [schwefel221, schwefel221_object]:
            for ndim in range(1, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_schwefel221(ndim - 1)))
            self.assertTrue(sample.check_origin(func))

    def test_rosenbrock(self):
        sample = TestCases(is_shifted=True, is_rotated=True)
        rosenbrock_object = Rosenbrock()
        for func in [rosenbrock, rosenbrock_object]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_rosenbrock(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))

    def test_schwefel12(self):
        sample = TestCases(is_shifted=True, is_rotated=True)
        schwefel12_object = Schwefel12()
        for func in [schwefel12, schwefel12_object]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_schwefel12(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))


if __name__ == '__main__':
    unittest.main()
