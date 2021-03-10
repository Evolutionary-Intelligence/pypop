from unittest import TestCase
import numpy as np

from utils import fitness_shaping


class Test(TestCase):
    def test_fitness_shaping(self):
        y = np.arange(5)
        self.assertTrue(np.allclose(fitness_shaping(y),
                                    [0.43704257, 0.08457026, -0.12161283, -0.2, -0.2]))
        y = np.arange(7)
        self.assertTrue(np.allclose(fitness_shaping(y),
                                    [0.38707304, 0.14285714, 0., -0.10135876, -0.14285714, -0.14285714, -0.14285714]))
