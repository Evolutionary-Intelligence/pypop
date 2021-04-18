import unittest

from benchmarks.base_functions import _squeeze_and_check
from benchmarks.base_functions import *
from test_cases import TestCases


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
        sphere_object = Sphere()
        for func in [sphere, sphere_object]:
            x1 = [4, 1, 0, 1, 4]
            self.assertTrue(sample.compare(func, 1, x1))
            x2 = [8, 2, 0, 2, 8]
            self.assertTrue(sample.compare(func, 2, x2))
            x3 = [12, 3, 0, 3, 12]
            self.assertTrue(sample.compare(func, 3, x3))
            x4 = [0, 4, 4, 4, 30, 30, 30]
            self.assertTrue(sample.compare(func, 4, x4))
            x5 = [0, 5, 5, 5, 55, 55, 55]
            self.assertTrue(sample.compare(func, 5, x5))
            x6 = [0, 6, 6, 6, 91, 91, 91]
            self.assertTrue(sample.compare(func, 6, x6))
            x7 = [0, 7, 7, 7, 140, 140, 140, 91]
            self.assertTrue(sample.compare(func, 7, x7))
            self.assertTrue(sample.check_origin(func))

    def test_cigar(self):
        sample = TestCases()
        cigar_object = Cigar()
        for func in [cigar, cigar_object]:
            x2 = [4000004, 1000001, 0, 1000001, 4000004]
            self.assertTrue(sample.compare(func, 2, x2))
            x3 = [8000004, 2000001, 0, 2000001, 8000004]
            self.assertTrue(sample.compare(func, 3, x3))
            x4 = [0, 3000001, 3000001, 3000001, 29000001, 29000001, 14000016]
            self.assertTrue(sample.compare(func, 4, x4))
            x5 = [0, 4000001, 4000001, 4000001, 54000001, 54000001, 30000025]
            self.assertTrue(sample.compare(func, 5, x5))
            x6 = [0, 5000001, 5000001, 5000001, 90000001, 90000001, 55000036]
            self.assertTrue(sample.compare(func, 6, x6))
            x7 = [0, 6000001, 6000001, 6000001, 139000001, 139000001, 91000049, 91000000]
            self.assertTrue(sample.compare(func, 7, x7))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_discus(self):
        sample = TestCases()
        discus_object = Discus()
        for func in [discus, discus_object]:
            x2 = [4000004, 1000001, 0, 1000001, 4000004]
            self.assertTrue(sample.compare(func, 2, x2))
            x3 = [4000008, 1000002, 0, 1000002, 4000008]
            self.assertTrue(sample.compare(func, 3, x3))
            x4 = [0, 1000003, 1000003, 1000003, 1000029, 1000029, 16000014]
            self.assertTrue(sample.compare(func, 4, x4))
            x5 = [0, 1000004, 1000004, 1000004, 1000054, 1000054, 25000030]
            self.assertTrue(sample.compare(func, 5, x5))
            x6 = [0, 1000005, 1000005, 1000005, 1000090, 1000090, 36000055]
            self.assertTrue(sample.compare(func, 6, x6))
            x7 = [0, 1000006, 1000006, 1000006, 1000139, 1000139, 49000091, 91]
            self.assertTrue(sample.compare(func, 7, x7))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_cigar_discus(self):
        sample = TestCases()
        cigar_discus_object = CigarDiscus()
        for func in [cigar_discus, cigar_discus_object]:
            x2 = [4080004, 1020001, 0, 1020001, 4080004]
            self.assertTrue(sample.compare(func, 2, x2))
            x3 = [4040004, 1010001, 0, 1010001, 4040004]
            self.assertTrue(sample.compare(func, 3, x3))
            x4 = [0, 1020001, 1020001, 1020001, 16130001, 16130001, 1130016]
            self.assertTrue(sample.compare(func, 4, x4))
            x5 = [0, 1030001, 1030001, 1030001, 25290001, 25290001, 1290025]
            self.assertTrue(sample.compare(func, 5, x5))
            x6 = [0, 1040001, 1040001, 1040001, 36540001, 36540001, 1540036]
            self.assertTrue(sample.compare(func, 6, x6))
            x7 = [0, 1050001, 1050001, 1050001, 49900001, 49900001, 1900049, 36550000]
            self.assertTrue(sample.compare(func, 7, x7))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_ellipsoid(self):
        sample = TestCases()
        ellipsoid_object = Ellipsoid()
        for func in [ellipsoid, ellipsoid_object]:
            x2 = [4000004, 1000001, 0, 1000001, 4000004]
            self.assertTrue(sample.compare(func, 2, x2))
            x3 = [4004004, 1001001, 0, 1001001, 4004004]
            self.assertTrue(sample.compare(func, 3, x3))
            x4 = [0, 1010101, 1010101, 1010101, 16090401, 16090401, 1040916]
            self.assertTrue(sample.compare(func, 4, x4))
            x5 = [0, 1032655, 1032655, 1032655, 25515092, 25515092, 1136022]
            self.assertTrue(sample.compare(func, 5, x5))
            x6 = [0, 1067345, 1067345, 1067345, 37643416, 37643416, 1292664]
            self.assertTrue(sample.compare(func, 6, x6))
            x7 = [0, 1111111, 1111111, 1111111, 52866941, 52866941, 1508909, 38669410]
            self.assertTrue(sample.compare(func, 7, x7))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_different_powers(self):
        sample = TestCases()
        different_powers_object = DifferentPowers()
        for func in [different_powers, different_powers_object]:
            x2 = [68, 2, 0, 2, 68]
            self.assertTrue(sample.compare(func, 2, x2))
            x3 = [84, 3, 0, 3, 84]
            self.assertTrue(sample.compare(func, 3, x3))
            x4 = [0, 4, 4, 4, 4275.6, 4275.6, 81.3]
            self.assertTrue(sample.compare(func, 4, x4, 0.1))
            x5 = [0, 5, 5, 5, 16739, 16739, 203]
            self.assertTrue(sample.compare(func, 5, x5))
            x6 = [0, 6, 6, 6, 51473.5, 51473.5, 437.1]
            self.assertTrue(sample.compare(func, 6, x6, 0.1))
            x7 = [0, 7, 7, 7, 133908.7, 133908.7, 847.4, 52736.8]
            self.assertTrue(sample.compare(func, 7, x7, 0.1))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))

    def test_schwefel221(self):
        sample = TestCases()
        schwefel221_object = Schwefel221()
        for func in [schwefel221, schwefel221_object]:
            x1 = [2, 1, 0, 1, 2]
            self.assertTrue(sample.compare(func, 1, x1))
            x2 = [2, 1, 0, 1, 2]
            self.assertTrue(sample.compare(func, 2, x2))
            x3 = [2, 1, 0, 1, 2]
            self.assertTrue(sample.compare(func, 3, x3))
            x4 = [0, 1, 1, 1, 4, 4, 4]
            self.assertTrue(sample.compare(func, 4, x4))
            x5 = [0, 1, 1, 1, 5, 5, 5]
            self.assertTrue(sample.compare(func, 5, x5))
            x6 = [0, 1, 1, 1, 6, 6, 6]
            self.assertTrue(sample.compare(func, 6, x6))
            x7 = [0, 1, 1, 1, 7, 7, 7, 6]
            self.assertTrue(sample.compare(func, 7, x7))
            self.assertTrue(sample.check_origin(func))

    def test_rosenbrock(self):
        sample = TestCases()
        rosenbrock_object = Rosenbrock()
        for func in [rosenbrock, rosenbrock_object]:
            x2 = [409, 4, 1, 0, 401]
            self.assertTrue(sample.compare(func, 2, x2))
            x3 = [810, 4, 2, 400, 4002]
            self.assertTrue(sample.compare(func, 3, x3))
            x4 = [3, 0, 1212, 804, 2705, 17913, 24330]
            self.assertTrue(sample.compare(func, 4, x4))
            x5 = [4, 0, 1616, 808, 14814, 30038, 68450]
            self.assertTrue(sample.compare(func, 5, x5))
            x6 = [5, 0, 2020, 808, 50930, 126154, 164579]
            self.assertTrue(sample.compare(func, 6, x6))
            x7 = [6, 0, 2424, 1208, 135055, 210303, 349519, 51031]
            self.assertTrue(sample.compare(func, 7, x7))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))

    def test_schwefel12(self):
        sample = TestCases()
        schwefel12_object = Schwefel12()
        for func in [schwefel12, schwefel12_object]:
            x2 = [4, 1, 0, 5, 20]
            self.assertTrue(sample.compare(func, 2, x2))
            x3 = [8, 2, 0, 6, 24]
            self.assertTrue(sample.compare(func, 3, x3))
            x4 = [0, 30, 30, 2, 146, 10, 18]
            self.assertTrue(sample.compare(func, 4, x4))
            x5 = [0, 55, 55, 3, 371, 19, 55]
            self.assertTrue(sample.compare(func, 5, x5))
            x6 = [0, 91, 91, 7, 812, 28, 195]
            self.assertTrue(sample.compare(func, 6, x6))
            x7 = [0, 140, 140, 8, 1596, 44, 564, 812]
            self.assertTrue(sample.compare(func, 7, x7))
            with self.assertRaisesRegex(TypeError, 'The size should > 1+'):
                sample.compare(func, 1, np.empty((5,)))
            self.assertTrue(sample.check_origin(func))


if __name__ == '__main__':
    unittest.main()
