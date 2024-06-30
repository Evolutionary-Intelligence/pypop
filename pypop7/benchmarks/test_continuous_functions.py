import unittest

from pypop7.benchmarks.base_functions import sphere as base_sphere
from pypop7.benchmarks.continuous_functions import *
from pypop7.benchmarks.cases import *


class Test(unittest.TestCase):
    def test_load_shift_and_rotation(self):
        func, ndim, seed = base_sphere, 2, 1
        generate_shift_vector(func, 2, [-1.0, -2.0], [1.0, 2.0], 0)
        generate_rotation_matrix(func, ndim, seed)
        shift_vector, rotation_matrix = load_shift_and_rotation(func, [0, 0])
        self.assertTrue(np.allclose(shift_vector, [2.739233746429086125e-01, -9.208531449445187533e-01]))
        self.assertTrue(np.allclose(rotation_matrix, [[7.227690004350708630e-01, 6.910896989610599839e-01],
                                                      [6.910896989610598729e-01, -7.227690004350709740e-01]]))

    def test_sphere(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [sphere, Sphere()]:
            for ndim in range(1, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_sphere(ndim - 1)))
            self.assertTrue(sample.check_origin(func))

    def test_cigar(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [cigar, Cigar()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_cigar(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_discus(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [discus, Discus()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_discus(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_cigar_discus(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [cigar_discus, CigarDiscus()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_cigar_discus(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_ellipsoid(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [ellipsoid, Ellipsoid()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_ellipsoid(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_different_powers(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [different_powers, DifferentPowers()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_different_powers(ndim - 2), atol=0.1))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_schwefel221(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [schwefel221, Schwefel221()]:
            for ndim in range(1, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_schwefel221(ndim - 1)))
            self.assertTrue(sample.check_origin(func))

    def test_step(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [step, Step()]:
            for ndim in range(1, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_step(ndim - 1)))
            self.assertTrue(sample.check_origin(func))

    def test_schwefel222(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [schwefel222, Schwefel222()]:
            for ndim in range(1, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_schwefel222(ndim - 1)))
            self.assertTrue(sample.check_origin(func))

    def test_rosenbrock(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [rosenbrock, Rosenbrock()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_rosenbrock(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))

    def test_schwefel12(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [schwefel12, Schwefel12()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_schwefel12(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_exponential(self):
        for func in [exponential, Exponential()]:
            for ndim in range(1, 8):
                x = np.zeros((ndim,))
                generate_shift_vector(func, ndim, -np.ones((ndim,)), 2 * np.ones((ndim,)), 2021 + ndim)
                generate_rotation_matrix(func, ndim, ndim)
                shift_vector, rotation_matrix = load_shift_and_rotation(func, x)
                x = np.dot(np.linalg.inv(rotation_matrix), x)
                x += shift_vector
                self.assertTrue(np.abs(func(x) + 1) < 1e-9)

    def test_griewank(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [griewank, Griewank()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_griewank(ndim - 2), atol=0.001))
            self.assertTrue(sample.check_origin(func))

    def test_bohachevsky(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [bohachevsky, Bohachevsky()]:
            for ndim in range(1, 5):
                self.assertTrue(sample.compare(func, ndim, get_y_bohachevsky(ndim - 1), atol=0.1))
            self.assertTrue(sample.check_origin(func))

    def test_ackley(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [ackley, Ackley()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_ackley(ndim - 2), atol=0.001))
            self.assertTrue(sample.check_origin(func))

    def test_rastrigin(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [rastrigin, Rastrigin()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_rastrigin(ndim - 2)))
            self.assertTrue(sample.check_origin(func))

    def test_scaled_rastrigin(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [scaled_rastrigin, ScaledRastrigin()]:
            for ndim in range(1, 4):
                self.assertTrue(sample.compare(func, ndim, get_y_scaled_rastrigin(ndim - 1), atol=0.01))
            self.assertTrue(sample.check_origin(func))

    def test_skew_rastrigin(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [skew_rastrigin, SkewRastrigin()]:
            for ndim in range(1, 5):
                self.assertTrue(sample.compare(func, ndim, get_y_skew_rastrigin(ndim - 1), atol=0.1))
            self.assertTrue(sample.check_origin(func))

    def test_levy_montalvo(self):
        for func in [levy_montalvo, LevyMontalvo()]:
            for ndim in range(1, 8):
                x = -np.ones((ndim,))
                generate_shift_vector(func, ndim, -np.ones((ndim,)), 3 * np.ones((ndim,)), 2021 + ndim)
                generate_rotation_matrix(func, ndim, ndim)
                shift_vector, rotation_matrix = load_shift_and_rotation(func, x)
                x = np.dot(np.linalg.inv(rotation_matrix), x)
                x += shift_vector
                self.assertTrue(np.abs(func(x)) < 1e-9)

    def test_michalewicz(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [michalewicz, Michalewicz()]:
            self.assertTrue(sample.check_origin(func))

    def test_salomon(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [salomon, Salomon()]:
            self.assertTrue(sample.check_origin(func))

    def test_shubert(self):
        for func in [shubert, Shubert()]:
            generate_rotation_matrix(func, 2, 2)
            generate_shift_vector(func, 2, -7.0 * np.ones((2,)), 5.0 * np.ones((2,)), 2021)
            for minimizer in get_y_shubert():
                shift_vector, rotation_matrix = load_shift_and_rotation(func, minimizer)
                minimizer = np.dot(np.linalg.inv(rotation_matrix), minimizer)
                minimizer += shift_vector
                self.assertTrue((np.abs(func(minimizer) + 186.7309) < 1e-3))

    def test_schaffer(self):
        sample = Cases(is_shifted=True, is_rotated=True)
        for func in [schaffer, Schaffer()]:
            for ndim in range(1, 4):
                self.assertTrue(sample.compare(func, ndim, get_y_schaffer(ndim - 1), atol=0.01))
            self.assertTrue(sample.check_origin(func))


if __name__ == '__main__':
    unittest.main()
