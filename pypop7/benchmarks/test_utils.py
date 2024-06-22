"""Only for the testing purpose. Online documentation:
    https://pypop.readthedocs.io/en/latest/utils.html
"""
import unittest

import numpy as np  # engine for numerical computing

from pypop7.benchmarks import base_functions as bf
from pypop7.benchmarks import rotated_functions as rf
from pypop7.benchmarks.utils import generate_xyz
from pypop7.benchmarks.utils import plot_contour
from pypop7.benchmarks.utils import plot_surface


# test function for plotting
def cd(x):  # from https://arxiv.org/pdf/1610.00040v1.pdf
    return 7.0 * (x[0] ** 2) + 6.0 * x[0] * x[1] + 8.0 * (x[1] ** 2)


class TestUtils(unittest.TestCase):
    def test_generate_xyz(self):
        x, y, z = generate_xyz(bf.sphere, [0.0, 1.0], [0.0, 1.0], num=2)
        self.assertTrue(np.allclose(x, np.array([[0.0, 1.0], [0.0, 1.0]])))
        self.assertTrue(np.allclose(y, np.array([[0.0, 0.0], [1.0, 1.0]])))
        self.assertTrue(np.allclose(z, np.array([[0.0, 1.0], [1.0, 2.0]])))

    def test_plot_contour(self):
        # plot smoothness and convexity
        plot_contour(bf.sphere, [-10.0, 10.0], [-10.0, 10.0])
        plot_contour(bf.ellipsoid, [-10.0, 10.0], [-10.0, 10.0])
        # plot multi-modality
        plot_contour(bf.rastrigin, [-10.0, 10.0], [-10.0, 10.0])
        # plot non-convexity
        plot_contour(bf.ackley, [-10.0, 10.0], [-10.0, 10.0])
        # plot non-separability
        plot_contour(cd, [-10.0, 10.0], [-10.0, 10.0])
        # plot ill-condition and non-separability
        rf.generate_rotation_matrix(rf.ellipsoid, 2, 72)
        plot_contour(rf.ellipsoid, [-10.0, 10.0], [-10.0, 10.0], 7)

    def test_plot_surface(self):
        # plot smoothness and convexity
        plot_surface(bf.sphere, [-10.0, 10.0], [-10.0, 10.0])
        plot_surface(bf.ellipsoid, [-10.0, 10.0], [-10.0, 10.0])
        # plot multi-modality
        plot_surface(bf.rastrigin, [-10.0, 10.0], [-10.0, 10.0])
        # plot non-convexity
        plot_surface(bf.ackley, [-10.0, 10.0], [-10.0, 10.0])
        # plot non-separability
        plot_surface(cd, [-10.0, 10.0], [-10.0, 10.0])
        # plot ill-condition and non-separability
        rf.generate_rotation_matrix(rf.ellipsoid, 2, 72)
        plot_surface(rf.ellipsoid, [-10.0, 10.0], [-10.0, 10.0], 7)


if __name__ == '__main__':
    unittest.main()
