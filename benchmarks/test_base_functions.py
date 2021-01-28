import unittest
import numpy as np

from base_functions import _squeeze_and_check


class TestBaseFunctions(unittest.TestCase):
    def test_squeeze_and_check(self):
        self.assertEqual(_squeeze_and_check(0), np.array([0]))
        self.assertEqual(_squeeze_and_check(np.array(0)), np.array([0]))
        x1 = np.array([0.7])
        self.assertEqual(_squeeze_and_check(x1), x1)
        x2 = np.array([0.0, 1.0])
        self.assertTrue(np.allclose(_squeeze_and_check(x2), x2))
        x3 = np.arange(6).reshape(2, 3)
        with self.assertRaisesRegex(TypeError, "The number+"):
            _squeeze_and_check(x3)
        with self.assertRaisesRegex(TypeError, "The size should > 1+"):
            _squeeze_and_check(x1, True)
        with self.assertRaisesRegex(TypeError, "the size should != 0."):
            _squeeze_and_check([])


if __name__ == '__main__':
    unittest.main()
