import unittest

from pypop7.benchmarks.base_functions import sphere
from pypop7.benchmarks.base_functions import step
# test the correctness of benchmarking functions via sampling (test cases)
from pypop7.benchmarks.cases import *


class TestBaseFunctions(unittest.TestCase):
    def test_squeeze_and_check(self):
        self.assertEqual(squeeze_and_check(0), np.array([0]))
        self.assertEqual(squeeze_and_check(np.array(0)), np.array([0]))
        x1 = np.array([0.7])
        self.assertEqual(squeeze_and_check(x1), x1)
        x2 = np.array([0.0, 1.0])
        self.assertTrue(np.allclose(squeeze_and_check(x2), x2))
        x3 = np.arange(6).reshape(2, 3)
        with self.assertRaisesRegex(TypeError, 'The number+'):
            squeeze_and_check(x3)
        with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
            squeeze_and_check(x1, True)
        with self.assertRaisesRegex(TypeError, 'the size should != 0.'):
            squeeze_and_check([])

    def test_cigar(self):
        sample = Cases()
        for func in [cigar, Cigar()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_cigar(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_discus(self):
        sample = Cases()
        for func in [discus, Discus()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_discus(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_cigar_discus(self):
        sample = Cases()
        for func in [cigar_discus, CigarDiscus()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_cigar_discus(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_ellipsoid(self):
        sample = Cases()
        for func in [ellipsoid, Ellipsoid()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_ellipsoid(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_different_powers(self):
        sample = Cases()
        for func in [different_powers, DifferentPowers()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_different_powers(ndim - 2), atol=0.1))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_schwefel221(self):
        sample = Cases()
        for func in [schwefel221, Schwefel221()]:
            for ndim in range(1, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_schwefel221(ndim - 1)))
            self.assertTrue(sample.check_origin(func))

    def test_schwefel222(self):
        sample = Cases()
        for func in [schwefel222, Schwefel222()]:
            for ndim in range(1, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_schwefel222(ndim - 1)))
            self.assertTrue(sample.check_origin(func))

    def test_rosenbrock(self):
        sample = Cases()
        for func in [rosenbrock, Rosenbrock()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_rosenbrock(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))

    def test_schwefel12(self):
        sample = Cases()
        for func in [schwefel12, Schwefel12()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_schwefel12(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_exponential(self):
        for func in [exponential, Exponential()]:
            for ndim in range(1, 8):
                self.assertTrue(np.abs(func(np.zeros((ndim,))) + 1) < 1e-9)

    def test_griewank(self):
        sample = Cases()
        for func in [griewank, Griewank()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_griewank(ndim - 2), atol=0.001))
            self.assertTrue(sample.check_origin(func))

    def test_bohachevsky(self):
        sample = Cases()
        for func in [bohachevsky, Bohachevsky()]:
            for ndim in range(1, 5):
                self.assertTrue(sample.compare(func, ndim, get_y_bohachevsky(ndim - 1), atol=0.1))
            self.assertTrue(sample.check_origin(func))

    def test_ackley(self):
        sample = Cases()
        for func in [ackley, Ackley()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_ackley(ndim - 2), atol=0.001))
            self.assertTrue(sample.check_origin(func))

    def test_rastrigin(self):
        sample = Cases()
        for func in [rastrigin, Rastrigin()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_rastrigin(ndim - 2)))
            self.assertTrue(sample.check_origin(func))

    def test_scaled_rastrigin(self):
        sample = Cases()
        for func in [scaled_rastrigin, ScaledRastrigin()]:
            for ndim in range(1, 4):
                self.assertTrue(sample.compare(func, ndim, get_y_scaled_rastrigin(ndim - 1), atol=0.01))
            self.assertTrue(sample.check_origin(func))

    def test_levy_montalvo(self):
        for func in [levy_montalvo, LevyMontalvo()]:
            for ndim in range(1, 8):
                self.assertTrue(np.abs(func(-np.ones((ndim,)))) < 1e-9)

    def test_michalewicz(self):
        sample = Cases()
        for func in [michalewicz, Michalewicz()]:
            self.assertTrue(sample.check_origin(func))

    def test_salomon(self):
        sample = Cases()
        for func in [salomon, Salomon()]:
            self.assertTrue(sample.check_origin(func))

    def test_schaffer(self):
        sample = Cases()
        for func in [schaffer, Schaffer()]:
            for ndim in range(1, 4):
                self.assertTrue(sample.compare(func, ndim, get_y_schaffer(ndim - 1), atol=0.01))
            self.assertTrue(sample.check_origin(func))

    def test_shubert(self):
        for minimizer in get_y_shubert():
            self.assertTrue((np.abs(shubert(minimizer) + 186.7309) < 1e-3))

    def test_skew_rastrigin(self):
        sample = Cases()
        for ndim in range(1, 5):
            self.assertTrue(sample.compare(skew_rastrigin, ndim, get_y_skew_rastrigin(ndim - 1), atol=0.1))
        self.assertTrue(sample.check_origin(skew_rastrigin))

    def test_sphere(self):
        sample = Cases()
        for ndim in range(1, 8):
            self.assertTrue(sample.compare(sphere, ndim, get_y_sphere(ndim - 1)))
        self.assertTrue(sample.check_origin(sphere))

    def test_step(self):
        sample = Cases()
        for ndim in range(1, 8):
            self.assertTrue(sample.compare(step, ndim, get_y_step(ndim - 1)))
        self.assertTrue(sample.check_origin(step))


if __name__ == '__main__':
    unittest.main()
