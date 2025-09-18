import unittest

from pypop7.benchmarks.class_of_base_functions import Ackley
from pypop7.benchmarks.class_of_base_functions import Cigar
from pypop7.benchmarks.class_of_base_functions import CigarDiscus
from pypop7.benchmarks.class_of_base_functions import DifferentPowers
from pypop7.benchmarks.class_of_base_functions import Discus
from pypop7.benchmarks.class_of_base_functions import Ellipsoid
from pypop7.benchmarks.class_of_base_functions import Exponential
from pypop7.benchmarks.class_of_base_functions import Griewank
from pypop7.benchmarks.class_of_base_functions import LevyMontalvo
from pypop7.benchmarks.class_of_base_functions import Michalewicz
from pypop7.benchmarks.class_of_base_functions import Rastrigin
from pypop7.benchmarks.class_of_base_functions import Rosenbrock
from pypop7.benchmarks.class_of_base_functions import Salomon
from pypop7.benchmarks.class_of_base_functions import ScaledRastrigin
from pypop7.benchmarks.class_of_base_functions import Schaffer
from pypop7.benchmarks.class_of_base_functions import Schwefel12
from pypop7.benchmarks.class_of_base_functions import Schwefel221
from pypop7.benchmarks.class_of_base_functions import Schwefel222
from pypop7.benchmarks.class_of_base_functions import Shubert
from pypop7.benchmarks.class_of_base_functions import SkewRastrigin
from pypop7.benchmarks.class_of_base_functions import Sphere
from pypop7.benchmarks.class_of_base_functions import Step
# test the coding correctness of benchmarking functions
# via limited sampling (test cases)
from pypop7.benchmarks.cases import *


class TestBaseFunctions(unittest.TestCase):
    def test_ackley(self):
        sample = Cases()
        ackley = Ackley()
        for ndim in range(2, 8):
            self.assertTrue(sample.compare(ackley, ndim, get_y_ackley(ndim - 2), atol=0.001))
        self.assertTrue(sample.check_origin(ackley))

    def test_cigar(self):
        sample = Cases()
        cigar = Cigar()
        for ndim in range(2, 8):
            self.assertTrue(sample.compare(cigar, ndim, get_y_cigar(ndim - 2)))
        with self.assertRaisesRegex(TypeError, 'PyPop7: size should+'):
            sample.compare(cigar, 1, np.empty((5,)))
        self.assertTrue(sample.check_origin(cigar))

    def test_cigar_discus(self):
        sample = Cases()
        cigar_discus = CigarDiscus()
        for ndim in range(2, 8):
            self.assertTrue(sample.compare(cigar_discus, ndim, get_y_cigar_discus(ndim - 2)))
        with self.assertRaisesRegex(TypeError, 'PyPop7: size should+'):
            sample.compare(cigar_discus, 1, np.empty((5,)))
        self.assertTrue(sample.check_origin(cigar_discus))

    def test_different_powers(self):
        sample = Cases()
        different_powers = DifferentPowers()
        for ndim in range(2, 8):
            self.assertTrue(sample.compare(different_powers, ndim, get_y_different_powers(ndim - 2), atol=0.1))
        with self.assertRaisesRegex(TypeError, 'PyPop7: size should+'):
            sample.compare(different_powers, 1, np.empty((5,)))
        self.assertTrue(sample.check_origin(different_powers))

    def test_discus(self):
        sample = Cases()
        discus = Discus()
        for ndim in range(2, 8):
            self.assertTrue(sample.compare(discus, ndim, get_y_discus(ndim - 2)))
        with self.assertRaisesRegex(TypeError, 'PyPop7: size should+'):
            sample.compare(discus, 1, np.empty((5,)))
        self.assertTrue(sample.check_origin(discus))

    def test_ellipsoid(self):
        sample = Cases()
        ellipsoid = Ellipsoid()
        for ndim in range(2, 8):
            self.assertTrue(sample.compare(ellipsoid, ndim, get_y_ellipsoid(ndim - 2)))
        with self.assertRaisesRegex(TypeError, 'PyPop7: size should+'):
            sample.compare(ellipsoid, 1, np.empty((5,)))
        self.assertTrue(sample.check_origin(ellipsoid))

    def test_exponential(self):
        exponential = Exponential()
        for ndim in range(1, 8):
            self.assertTrue(np.abs(exponential(np.zeros((ndim,))) + 1) < 1e-9)

    def test_griewank(self):
        sample = Cases()
        griewank = Griewank()
        for ndim in range(2, 8):
            self.assertTrue(sample.compare(griewank, ndim, get_y_griewank(ndim - 2), atol=0.001))
        self.assertTrue(sample.check_origin(griewank))

    def test_levy_montalvo(self):
        levy_montalvo = LevyMontalvo()
        for ndim in range(1, 8):
            self.assertTrue(np.abs(levy_montalvo(-np.ones((ndim,)))) < 1e-9)

    def test_michalewicz(self):
        sample = Cases()
        michalewicz = Michalewicz()
        self.assertTrue(sample.check_origin(michalewicz))

    def test_rastrigin(self):
        sample = Cases()
        rastrigin = Rastrigin()
        for ndim in range(2, 8):
            self.assertTrue(sample.compare(rastrigin, ndim, get_y_rastrigin(ndim - 2)))
        self.assertTrue(sample.check_origin(rastrigin))

    def test_rosenbrock(self):
        sample = Cases()
        rosenbrock = Rosenbrock()
        for ndim in range(2, 8):
            self.assertTrue(sample.compare(rosenbrock, ndim, get_y_rosenbrock(ndim - 2)))
        with self.assertRaisesRegex(TypeError, 'PyPop7: size should+'):
            sample.compare(rosenbrock, 1, np.empty((5,)))

    def test_salomon(self):
        sample = Cases()
        salomon = Salomon()
        self.assertTrue(sample.check_origin(salomon))

    def test_scaled_rastrigin(self):
        sample = Cases()
        scaled_rastrigin = ScaledRastrigin()
        for ndim in range(1, 4):
            self.assertTrue(sample.compare(scaled_rastrigin, ndim, get_y_scaled_rastrigin(ndim - 1), atol=0.01))
        self.assertTrue(sample.check_origin(scaled_rastrigin))

    def test_schaffer(self):
        sample = Cases()
        schaffer = Schaffer()
        for ndim in range(1, 4):
            self.assertTrue(sample.compare(schaffer, ndim, get_y_schaffer(ndim - 1), atol=0.01))
        self.assertTrue(sample.check_origin(schaffer))

    def test_schwefel12(self):
        sample = Cases()
        schwefel12 = Schwefel12()
        for ndim in range(2, 8):
            self.assertTrue(sample.compare(schwefel12, ndim, get_y_schwefel12(ndim - 2)))
        with self.assertRaisesRegex(TypeError, 'PyPop7: size should+'):
            sample.compare(schwefel12, 1, np.empty((5,)))
        self.assertTrue(sample.check_origin(schwefel12))

    def test_schwefel221(self):
        sample = Cases()
        schwefel221 = Schwefel221()
        for ndim in range(1, 8):
            self.assertTrue(sample.compare(schwefel221, ndim, get_y_schwefel221(ndim - 1)))
        self.assertTrue(sample.check_origin(schwefel221))

    def test_schwefel222(self):
        sample = Cases()
        schwefel222 = Schwefel222()
        for ndim in range(1, 8):
            self.assertTrue(sample.compare(schwefel222, ndim, get_y_schwefel222(ndim - 1)))
        self.assertTrue(sample.check_origin(schwefel222))

    def test_shubert(self):
        shubert = Shubert()
        for minimizer in get_y_shubert():
            self.assertTrue((np.abs(shubert(minimizer) + 186.7309) < 1e-3))

    def test_skew_rastrigin(self):
        sample = Cases()
        skew_rastrigin = SkewRastrigin()
        for ndim in range(1, 5):
            self.assertTrue(sample.compare(skew_rastrigin, ndim, get_y_skew_rastrigin(ndim - 1), atol=0.1))
        self.assertTrue(sample.check_origin(skew_rastrigin))

    def test_sphere(self):
        sample = Cases()
        sphere = Sphere()
        for ndim in range(1, 8):
            self.assertTrue(sample.compare(sphere, ndim, get_y_sphere(ndim - 1)))
        self.assertTrue(sample.check_origin(sphere))

    def test_step(self):
        sample = Cases()
        step = Step()
        for ndim in range(1, 8):
            self.assertTrue(sample.compare(step, ndim, get_y_step(ndim - 1)))
        self.assertTrue(sample.check_origin(step))


if __name__ == '__main__':
    unittest.main()
