from unittest import TestCase
import numpy as np

import base_functions
from utils import _generate_xyz


class Test(TestCase):
    def test_generate_xyz(self):
        x, y, z = _generate_xyz(base_functions.sphere, [0, 1], [0, 1], num=2)
        self.assertTrue(np.allclose(x, np.array([[0., 1.], [0., 1.]])))
        self.assertTrue(np.allclose(y, np.array([[0., 0.], [1., 1.]])))
        self.assertTrue(np.allclose(z, np.array([[0., 1.], [1., 2.]])))
