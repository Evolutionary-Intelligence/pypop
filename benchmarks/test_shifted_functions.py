import unittest
import numpy as np

from base_functions import sphere as base_sphere
from shifted_functions import _generate_shift_vector, _load_shift_vector


class TestShiftedFunctions(unittest.TestCase):
    def test_generate_shift_vector(self):
        shift_vector = _generate_shift_vector('sphere', 2, -5, 10, 0)
        self.assertEqual(shift_vector.size, 2)
        self.assertTrue(np.all(shift_vector >= -5))
        self.assertTrue(np.all(shift_vector < 10))
        self.assertTrue(np.allclose(shift_vector, [4.554425309821814594e+00, -9.531992935419451030e-01]))

        shift_vector = _generate_shift_vector(base_sphere, 2, [-1, -2], [1, 2], 0)
        self.assertEqual(shift_vector.size, 2)
        self.assertTrue(np.all(shift_vector >= [-1, -2]))
        self.assertTrue(np.all(shift_vector < [1, 2]))
        self.assertTrue(np.allclose(shift_vector, [2.739233746429086125e-01, -9.208531449445187533e-01]))

        shift_vector = _generate_shift_vector(base_sphere, 1, -100, 100, 7)
        self.assertEqual(shift_vector.size, 1)
        self.assertTrue(np.all(shift_vector >= -100))
        self.assertTrue(np.all(shift_vector < 100))
        self.assertTrue(np.allclose(shift_vector, 2.501909332093339344e+01))

    def test_load_shift_vector(self):
        func = base_sphere
        _generate_shift_vector(func, 2, [-1, -2], [1, 2], 0)
        shift_vector = _load_shift_vector(func, [0, 0])
        self.assertTrue(np.allclose(shift_vector, [2.739233746429086125e-01, -9.208531449445187533e-01]))

        _generate_shift_vector(func, 3, -100, 100, 7)
        shift_vector = _load_shift_vector(func, [0, 0, 0])
        self.assertTrue(np.allclose(shift_vector,
                                    [2.501909332093339344e+01, 7.944276019391509180e+01, 5.513713804903869686e+01]))

        shift_vector = _load_shift_vector(func, 0, 77)
        self.assertTrue(np.allclose(shift_vector, 77))


if __name__ == '__main__':
    unittest.main()
