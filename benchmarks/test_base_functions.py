import unittest

from benchmarks.base_functions import _squeeze_and_check
from benchmarks.base_functions import *
from test_cases import *


class TestBaseFunctions(unittest.TestCase):
    def test_squeeze_and_check(self):
        self.assertEqual(_squeeze_and_check(0), np.array([0]))
        self.assertEqual(_squeeze_and_check(np.array(0)), np.array([0]))
        x1 = np.array([0.7])
        self.assertEqual(_squeeze_and_check(x1), x1)
        x2 = np.array([0.0, 1.0])
        self.assertTrue(np.allclose(_squeeze_and_check(x2), x2))
        x3 = np.arange(6).reshape(2, 3)
        with self.assertRaisesRegex(TypeError, 'The number+'):
            _squeeze_and_check(x3)
        with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
            _squeeze_and_check(x1, True)
        with self.assertRaisesRegex(TypeError, 'the size should != 0.'):
            _squeeze_and_check([])

    def test_sphere(self):
        sample = TestCases()
        for func in [sphere, Sphere()]:
            for ndim in range(1, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_sphere(ndim - 1)))
            self.assertTrue(sample.check_origin(func))

    def test_cigar(self):
        sample = TestCases()
        for func in [cigar, Cigar()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_cigar(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_discus(self):
        sample = TestCases()
        for func in [discus, Discus()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_discus(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_cigar_discus(self):
        sample = TestCases()
        for func in [cigar_discus, CigarDiscus()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_cigar_discus(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_ellipsoid(self):
        sample = TestCases()
        for func in [ellipsoid, Ellipsoid()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_ellipsoid(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_different_powers(self):
        sample = TestCases()
        for func in [different_powers, DifferentPowers()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_different_powers(ndim - 2), atol=0.1))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_schwefel221(self):
        sample = TestCases()
        for func in [schwefel221, Schwefel221()]:
            for ndim in range(1, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_schwefel221(ndim - 1)))
            self.assertTrue(sample.check_origin(func))

    def test_step(self):
        sample = TestCases()
        for func in [step, Step()]:
            for ndim in range(1, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_step(ndim - 1)))
            self.assertTrue(sample.check_origin(func))

    def test_schwefel222(self):
        sample = TestCases()
        for func in [schwefel222, Schwefel222()]:
            for ndim in range(1, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_schwefel222(ndim - 1)))
            self.assertTrue(sample.check_origin(func))

    def test_rosenbrock(self):
        sample = TestCases()
        for func in [rosenbrock, Rosenbrock()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_rosenbrock(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))

    def test_schwefel12(self):
        sample = TestCases()
        for func in [schwefel12, Schwefel12()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_schwefel12(ndim - 2)))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_griewank(self):
        sample = TestCases()
        for func in [griewank, Griewank()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_griewank(ndim - 2), atol=0.001))
            self.assertTrue(sample.check_origin(func))

    def test_ackley(self):
        sample = TestCases()
        for func in [ackley, Ackley()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_ackley(ndim - 2), atol=0.001))
            self.assertTrue(sample.check_origin(func))

    def test_rastrigin(self):
        sample = TestCases()
        for func in [rastrigin, Rastrigin()]:
            for ndim in range(2, 8):
                self.assertTrue(sample.compare(func, ndim, get_y_rastrigin(ndim - 2)))
            self.assertTrue(sample.check_origin(func))


if __name__ == '__main__':
    unittest.main()
