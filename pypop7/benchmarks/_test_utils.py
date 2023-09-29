from unittest import TestCase
import numpy as np

from pypop7.benchmarks import base_functions
from pypop7.benchmarks.utils import generate_xyz, plot_contour


class Test(TestCase):
    def test_generate_xyz(self):
        x, y, z = generate_xyz(base_functions.sphere, [0, 1], [0, 1], num=2)
        self.assertTrue(np.allclose(x, np.array([[0., 1.], [0., 1.]])))
        self.assertTrue(np.allclose(y, np.array([[0., 0.], [1., 1.]])))
        self.assertTrue(np.allclose(z, np.array([[0., 1.], [1., 2.]])))

    def test_plot_contour(self):
        plot_contour(base_functions.sphere, [-10, 10], [-10, 10])
        plot_contour(base_functions.ellipsoid, [-10, 10], [-10, 10])
