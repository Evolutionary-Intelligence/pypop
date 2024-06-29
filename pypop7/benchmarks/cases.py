"""Only for the testing purpose. Online documentation:
    https://pypop.readthedocs.io/en/latest/benchmarks.html#test-classes-and-data
"""
import numpy as np  # engine for numerical computing

from pypop7.benchmarks.shifted_functions import generate_shift_vector, load_shift_vector
from pypop7.benchmarks.rotated_functions import generate_rotation_matrix, load_rotation_matrix


class Cases(object):
    """Test the correctness of benchmarking functions via sampling (test cases).
    """
    def __init__(self, is_shifted=False, is_rotated=False):
        """Initialize the settings of test cases for the benchmarking function with or
           without the shift and/or rotation operation.
        :param is_shifted: whether or not to generate data for shift/transform, `bool`.
        :param is_rotated: whether or not to generate data for rotation, `bool`.
        """
        self.is_shifted = is_shifted  # whether or not to generate data for shift
        self.is_rotated = is_rotated  # whether or not to generate data for rotation
        self.ndim = None  # number of dimensionality of all test cases

    def make_test_cases(self, ndim=None):
        """Make multiple test cases for a specific dimension (only ranged in [1, 7]).

        .. note:: The number of test cases may be different on different dimensions.

        :param ndim: number of dimensions (only ranged in [1, 7]), `int`.
        :return: `ndarray` of dtype `np.float64`, where each row is a test case.
        """
        self.ndim = ndim
        if ndim == 1:
            x = [[-2],
                 [-1],
                 [0],
                 [1],
                 [2]]
        elif ndim == 2:
            x = [[-2, 2],
                 [-1, 1],
                 [0, 0],
                 [1, 1],
                 [2, 2]]
        elif ndim == 3:
            x = [[-2, 2, 2],
                 [-1, 1, 1],
                 [0, 0, 0],
                 [1, 1, -1],
                 [2, 2, -2]]
        elif ndim == 4:
            x = [[0, 0, 0, 0],
                 [1, 1, 1, 1],
                 [-1, -1, -1, -1],
                 [1, -1, 1, -1],
                 [1, 2, 3, 4],
                 [1, -2, 3, -4],
                 [-4, 3, 2, -1]]
        elif ndim == 5:
            x = [[0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1],
                 [-1, -1, -1, -1, -1],
                 [1, -1, 1, -1, 1],
                 [1, 2, 3, 4, 5],
                 [1, -2, 3, -4, 5],
                 [-5, 4, 3, 2, -1]]
        elif ndim == 6:
            x = [[0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [-1, -1, -1, -1, -1, -1],
                 [1, -1, 1, -1, 1, 1],
                 [1, 2, 3, 4, 5, 6],
                 [1, -2, 3, -4, 5, -6],
                 [-6, 5, 4, 3, 2, -1]]
        elif ndim == 7:
            x = [[0, 0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1, 1],
                 [-1, -1, -1, -1, -1, -1, -1],
                 [1, -1, 1, -1, 1, 1, -1],
                 [1, 2, 3, 4, 5, 6, 7],
                 [1, -2, 3, -4, 5, -6, 7],
                 [-7, 6, 5, 4, 3, 2, -1],
                 [0, 1, 2, 3, 4, 5, 6]]
        else:
            raise TypeError('Number of dimensions should >= 1 and <= 7.')
        return np.array(x, dtype=np.float64)  # `ndarray` should be of dtype `np.float64`

    def compare(self, func, ndim, y_true, shift_vector=None, rotation_matrix=None, atol=1e-3):
        """Compare true function values with these returned by the used benchmark function.

        :param func: benchmarking function, `func`.
        :param ndim: number of dimensions (only ranged in [1, 7]), `int`.
        :param y_true: `ndarray`, where each element is the true function value of the corresponding test case.
        :param shift_vector: shift vector, `ndarray`.
        :param rotation_matrix: rotation matrix, `ndarray`.
        :param atol: absolute tolerance parameter, `float`.
        :return: `True` if all function values computed on test cases match `y_true`; otherwise, `False`.
        """
        x = self.make_test_cases(ndim)
        y = np.empty((x.shape[0],))
        for i in range(x.shape[0]):
            if self.is_rotated:
                generate_rotation_matrix(func, ndim, ndim)
                rotation_matrix = load_rotation_matrix(func, x[i], rotation_matrix)
                x[i] = np.dot(np.linalg.inv(rotation_matrix), x[i])
            if self.is_shifted:
                generate_shift_vector(func, ndim, -10.0 * np.ones((ndim,)), 7.0 * np.ones((ndim,)), 2021 + ndim)
                x[i] = x[i] + load_shift_vector(func, x[i], shift_vector)
            y[i] = func(x[i])
        return np.allclose(y, y_true, atol=atol)

    def check_origin(self, func, n_samples=7):
        """Check the origin point of which the function value is zero via random sampling (test cases).

        :param func: benchmarking function, `func`.
        :param n_samples: number of samples, `int`.
        :return: `True` if all function values computed on test cases are zeros, otherwise `False`; `bool`.
        """
        # note that here `7` or `77` is just the **egg** of this open-source Python library
        ndims = np.random.default_rng(77).integers(2, 100, size=(n_samples,))
        self.ndim = ndims
        is_zero = True
        for d in ndims:
            x = np.zeros((d,))
            if self.is_shifted:
                generate_shift_vector(func, d, -np.ones((d,)), np.ones((d,)), d)
                x += load_shift_vector(func, x)
            if self.is_rotated:
                generate_rotation_matrix(func, d, d)
            if np.abs(func(x)) > 1e-3:
                is_zero = False
                break
        return is_zero


# expected (true) function values for test cases given in Cases class
def get_y_sphere(ndim):
    """Get test data for **Sphere** test function.
    """
    y = [[4.0, 1.0, 0.0, 1.0, 4.0],
         [8.0, 2.0, 0.0, 2.0, 8.0],
         [12.0, 3.0, 0.0, 3.0, 12.0],
         [0.0, 4.0, 4.0, 4.0, 30.0, 30.0, 30.0],
         [0.0, 5.0, 5.0, 5.0, 55.0, 55.0, 55.0],
         [0.0, 6.0, 6.0, 6.0, 91.0, 91.0, 91.0],
         [0.0, 7.0, 7.0, 7.0, 140.0, 140.0, 140.0, 91.0]]
    return y[ndim]


def get_y_cigar(ndim):
    """Get test data for **Cigar** test function.
    """
    y = [[4000004.0, 1000001.0, 0.0, 1000001.0, 4000004.0],
         [8000004.0, 2000001.0, 0.0, 2000001.0, 8000004.0],
         [0.0, 3000001.0, 3000001.0, 3000001.0, 29000001.0, 29000001.0, 14000016.0],
         [0.0, 4000001.0, 4000001.0, 4000001.0, 54000001.0, 54000001.0, 30000025.0],
         [0.0, 5000001.0, 5000001.0, 5000001.0, 90000001.0, 90000001.0, 55000036.0],
         [0.0, 6000001.0, 6000001.0, 6000001.0, 139000001.0, 139000001.0, 91000049.0, 91000000.0]]
    return y[ndim]


def get_y_discus(ndim):
    """Get test data for **Discus** test function.
    """
    y = [[4000004, 1000001, 0, 1000001, 4000004],
         [4000008, 1000002, 0, 1000002, 4000008],
         [0, 1000003, 1000003, 1000003, 1000029, 1000029, 16000014],
         [0, 1000004, 1000004, 1000004, 1000054, 1000054, 25000030],
         [0, 1000005, 1000005, 1000005, 1000090, 1000090, 36000055],
         [0, 1000006, 1000006, 1000006, 1000139, 1000139, 49000091, 91]]
    return y[ndim]


def get_y_cigar_discus(ndim):
    """Get test data for **Cigar-Discus** test function.
    """
    y = [[4080004, 1020001, 0, 1020001, 4080004],
         [4040004, 1010001, 0, 1010001, 4040004],
         [0, 1020001, 1020001, 1020001, 16130001, 16130001, 1130016],
         [0, 1030001, 1030001, 1030001, 25290001, 25290001, 1290025],
         [0, 1040001, 1040001, 1040001, 36540001, 36540001, 1540036],
         [0, 1050001, 1050001, 1050001, 49900001, 49900001, 1900049, 36550000]]
    return y[ndim]


def get_y_ellipsoid(ndim):
    """Get test data for **Ellipsoid** test function.
    """
    y = [[4000004, 1000001, 0, 1000001, 4000004],
         [4004004, 1001001, 0, 1001001, 4004004],
         [0, 1010101, 1010101, 1010101, 16090401, 16090401, 1040916],
         [0, 1032655, 1032655, 1032655, 25515092, 25515092, 1136022],
         [0, 1067345, 1067345, 1067345, 37643416, 37643416, 1292664],
         [0, 1111111, 1111111, 1111111, 52866941, 52866941, 1508909, 38669410]]
    return y[ndim]


def get_y_different_powers(ndim):
    """Get test data for **Different-Powers** test function.
    """
    y = [[68, 2, 0, 2, 68],
         [84, 3, 0, 3, 84],
         [0, 4, 4, 4, 4275.6, 4275.6, 81.3],
         [0, 5, 5, 5, 16739, 16739, 203],
         [0, 6, 6, 6, 51473.5, 51473.5, 437.1],
         [0, 7, 7, 7, 133908.7, 133908.7, 847.4, 52736.8]]
    return y[ndim]


def get_y_schwefel221(ndim):
    """Get test data for **Schwefel221** test function.
    """
    y = [[2, 1, 0, 1, 2],
         [2, 1, 0, 1, 2],
         [2, 1, 0, 1, 2],
         [0, 1, 1, 1, 4, 4, 4],
         [0, 1, 1, 1, 5, 5, 5],
         [0, 1, 1, 1, 6, 6, 6],
         [0, 1, 1, 1, 7, 7, 7, 6]]
    return y[ndim]


def get_y_step(ndim):
    """Get test data for **Step** test function.
    """
    y = [[4, 1, 0, 1, 4],
         [8, 2, 0, 2, 8],
         [12, 3, 0, 3, 12],
         [0, 4, 4, 4, 30, 30, 30],
         [0, 5, 5, 5, 55, 55, 55],
         [0, 6, 6, 6, 91, 91, 91],
         [0, 7, 7, 7, 140, 140, 140, 91]]
    return y[ndim]


def get_y_schwefel222(ndim):
    """Get test data for **Schwefel222** test function.
    """
    y = [[4, 2, 0, 2, 4],
         [8, 3, 0, 3, 8],
         [14, 4, 0, 4, 14],
         [0, 5, 5, 5, 34, 34, 34],
         [0, 6, 6, 6, 135, 135, 135],
         [0, 7, 7, 7, 741, 741, 741],
         [0, 8, 8, 8, 5068, 5068, 5068, 21]]
    return y[ndim]


def get_y_rosenbrock(ndim):
    """Get test data for **Rosenbrock** test function.
    """
    y = [[409, 4, 1, 0, 401],
         [810, 4, 2, 400, 4002],
         [3, 0, 1212, 804, 2705, 17913, 24330],
         [4, 0, 1616, 808, 14814, 30038, 68450],
         [5, 0, 2020, 808, 50930, 126154, 164579],
         [6, 0, 2424, 1208, 135055, 210303, 349519, 51031]]
    return y[ndim]


def get_y_schwefel12(ndim):
    """Get test data for **Schwefel12** test function.
    """
    y = [[4, 1, 0, 5, 20],
         [8, 2, 0, 6, 24],
         [0, 30, 30, 2, 146, 10, 18],
         [0, 55, 55, 3, 371, 19, 55],
         [0, 91, 91, 7, 812, 28, 195],
         [0, 140, 140, 8, 1596, 44, 564, 812]]
    return y[ndim]


def get_y_griewank(ndim):
    """Get test data for **Griewank** test function.
    """
    y = [[1.066895, 0.589738, 0, 0.589738, 1.066895],
         [1.029230, 0.656567, 0, 0.656567, 1.029230],
         [0, 0.698951, 0.698951, 0.698951, 1.001870, 1.001870, 0.886208],
         [0, 0.728906, 0.728906, 0.728906, 1.017225, 1.017225, 0.992641],
         [0, 0.751538, 0.751538, 0.751538, 1.020074, 1.020074, 0.998490],
         [0, 0.769431, 0.769431, 0.769431, 1.037353, 1.037353, 1.054868, 1.024118]]
    return y[ndim]


def get_y_bohachevsky(ndim):
    """Get test data for **Bohachevsky** test function.
    """
    y = [[0, 0, 0, 0, 0],
         [12, 3.6, 0, 3.6, 12],
         [24, 7.2, 0, 7.2, 24],
         [0, 10.8, 10.8, 10.8, 73.2, 73.2, 57.6]]
    return y[ndim]


def get_y_ackley(ndim):
    """Get test data for **Ackley** test function.
    """
    y = [[6.593599, 3.625384, 0, 3.625384, 6.593599],
         [6.593599, 3.625384, 0, 3.625384, 6.593599],
         [0, 3.625384, 3.625384, 3.625384, 8.434694, 8.434694, 8.434694],
         [0, 3.625384, 3.625384, 3.625384, 9.697286, 9.697286, 9.697286],
         [0, 3.625384, 3.625384, 3.625384, 10.821680, 10.821680, 10.821680],
         [0, 3.625384, 3.625384, 3.625384, 11.823165, 11.823165, 11.823165, 10.275757]]
    return y[ndim]


def get_y_rastrigin(ndim):
    """Get test data for **Rastrigin** test function.
    """
    y = [[8, 2, 0, 2, 8],
         [12, 3, 0, 3, 12],
         [0, 4, 4, 4, 30, 30, 30],
         [0, 5, 5, 5, 55, 55, 55],
         [0, 6, 6, 6, 91, 91, 91],
         [0, 7, 7, 7, 140, 140, 140, 91]]
    return y[ndim]


def get_y_shubert():
    """Get test data for **Schaffer** test function.
    """
    minimizers = [[-7.0835, 4.858], [-7.0835, -7.7083], [-1.4251, -7.0835], [5.4828, 4.858],
                  [-1.4251, -0.8003], [4.858, 5.4828], [-7.7083, -7.0835], [-7.0835, -1.4251],
                  [-7.7083, -0.8003], [-7.7083, 5.4828], [-0.8003, -7.7083], [-0.8003, -1.4251],
                  [-0.8003, 4.8580], [-1.4251, 5.4828], [5.4828, -7.7083], [4.858, -7.0835],
                  [5.4828, -1.4251], [4.858, -0.8003]]
    return minimizers


def get_y_schaffer(ndim):
    """Get test data for **Schaffer** test function.
    """
    y = [[0.0, 0.0, 0.0, 0.0, 0.0],
         [3.220, 1.228, 0.0, 1.228, 3.220],
         [6.441, 2.456, 0.0, 2.456, 6.441]]
    return y[ndim]
