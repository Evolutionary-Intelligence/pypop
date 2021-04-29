import numpy as np

from benchmarks.shifted_functions import generate_shift_vector, _load_shift_vector
from benchmarks.rotated_functions import generate_rotation_matrix, _load_rotation_matrix


class TestCases(object):
    """Test correctness of benchmark functions via sampling (test cases).
    """
    def __init__(self, is_shifted=False, is_rotated=False):
        self.is_shifted = is_shifted
        self.is_rotated = is_rotated
        self.ndim = None

    def make_test_cases(self, ndim=None):
        """Make multiple test cases for a specific dimension ranged in [1, 7].

        Note that the number of test cases may be different for different dimensions.

        :param ndim: number of dimensions, an `int` scalar ranged in [1, 7].
        :return: a 2-d `ndarray` of dtype `np.float64`, where each row is a test case.
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
            raise TypeError('The number of dimensions should >=1 and <= 7.')
        return np.array(x, dtype=np.float64)

    def compare(self, func, ndim, y_true, shift_vector=None, rotation_matrix=None, atol=1e-08):
        """Compare true (expected) function values with these returned (computed) by benchmark function.

        :param func: benchmark function, a function object.
        :param ndim: number of dimensions, an `int` scalar ranged in [1, 7].
        :param y_true: a 1-d `ndarray`, where each element is the true function value of the corresponding test case.
        :param shift_vector: shift vector, a 1-d `ndarray`.
        :param rotation_matrix: rotation matrix, a 2-d `ndarray`.
        :param atol: absolute tolerance parameter, a `float` scalar.
        :return: `True` if all function values computed on test cases match `y_true`; otherwise, `False`.
        """
        x = self.make_test_cases(ndim)
        y = np.empty((x.shape[0],))
        for i in range(x.shape[0]):
            if self.is_rotated:
                generate_rotation_matrix(func, ndim, ndim)
                rotation_matrix = _load_rotation_matrix(func, x[i], rotation_matrix)
                x[i] = np.dot(np.linalg.inv(rotation_matrix), x[i])
            if self.is_shifted:
                generate_shift_vector(func, ndim, -10 * np.ones((ndim,)), 7 * np.ones((ndim,)), 2021 + ndim)
                x[i] = x[i] + _load_shift_vector(func, x[i], shift_vector)
            y[i] = func(x[i])
        return np.allclose(y, y_true, atol=atol)

    def check_origin(self, func, n_samples=7):
        """Check the origin point at which the function value is zero via random sampling (test cases).

        :param func: rotated function, a function object.
        :param n_samples: number of samples, an `int` scalar.
        :return: `True` if all function values computed on test cases are zeros; otherwise, `False`.
        """
        ndims = np.random.default_rng().integers(2, 100, size=(n_samples,))
        self.ndim = ndims
        is_zero = True
        for d in ndims:
            x = np.zeros((d,))
            if self.is_shifted:
                generate_shift_vector(func, d, -np.ones((d,)), np.ones((d,)), d)
                x += _load_shift_vector(func, x)
            if self.is_rotated:
                generate_rotation_matrix(func, d, d)
            if np.abs(func(x)) > 1e-9:
                is_zero = False
                break
        return is_zero


# expected (true) function values for test cases given in class TestCases
def get_y_sphere(ndim):
    y = [[4, 1, 0, 1, 4],
         [8, 2, 0, 2, 8],
         [12, 3, 0, 3, 12],
         [0, 4, 4, 4, 30, 30, 30],
         [0, 5, 5, 5, 55, 55, 55],
         [0, 6, 6, 6, 91, 91, 91],
         [0, 7, 7, 7, 140, 140, 140, 91]]
    return y[ndim]


def get_y_cigar(ndim):
    y = [[4000004, 1000001, 0, 1000001, 4000004],
         [8000004, 2000001, 0, 2000001, 8000004],
         [0, 3000001, 3000001, 3000001, 29000001, 29000001, 14000016],
         [0, 4000001, 4000001, 4000001, 54000001, 54000001, 30000025],
         [0, 5000001, 5000001, 5000001, 90000001, 90000001, 55000036],
         [0, 6000001, 6000001, 6000001, 139000001, 139000001, 91000049, 91000000]]
    return y[ndim]


def get_y_discus(ndim):
    y = [[4000004, 1000001, 0, 1000001, 4000004],
         [4000008, 1000002, 0, 1000002, 4000008],
         [0, 1000003, 1000003, 1000003, 1000029, 1000029, 16000014],
         [0, 1000004, 1000004, 1000004, 1000054, 1000054, 25000030],
         [0, 1000005, 1000005, 1000005, 1000090, 1000090, 36000055],
         [0, 1000006, 1000006, 1000006, 1000139, 1000139, 49000091, 91]]
    return y[ndim]


def get_y_cigar_discus(ndim):
    y = [[4080004, 1020001, 0, 1020001, 4080004],
         [4040004, 1010001, 0, 1010001, 4040004],
         [0, 1020001, 1020001, 1020001, 16130001, 16130001, 1130016],
         [0, 1030001, 1030001, 1030001, 25290001, 25290001, 1290025],
         [0, 1040001, 1040001, 1040001, 36540001, 36540001, 1540036],
         [0, 1050001, 1050001, 1050001, 49900001, 49900001, 1900049, 36550000]]
    return y[ndim]


def get_y_ellipsoid(ndim):
    y = [[4000004, 1000001, 0, 1000001, 4000004],
         [4004004, 1001001, 0, 1001001, 4004004],
         [0, 1010101, 1010101, 1010101, 16090401, 16090401, 1040916],
         [0, 1032655, 1032655, 1032655, 25515092, 25515092, 1136022],
         [0, 1067345, 1067345, 1067345, 37643416, 37643416, 1292664],
         [0, 1111111, 1111111, 1111111, 52866941, 52866941, 1508909, 38669410]]
    return y[ndim]


def get_y_different_powers(ndim):
    y = [[68, 2, 0, 2, 68],
         [84, 3, 0, 3, 84],
         [0, 4, 4, 4, 4275.6, 4275.6, 81.3],
         [0, 5, 5, 5, 16739, 16739, 203],
         [0, 6, 6, 6, 51473.5, 51473.5, 437.1],
         [0, 7, 7, 7, 133908.7, 133908.7, 847.4, 52736.8]]
    return y[ndim]


def get_y_schwefel221(ndim):
    y = [[2, 1, 0, 1, 2],
         [2, 1, 0, 1, 2],
         [2, 1, 0, 1, 2],
         [0, 1, 1, 1, 4, 4, 4],
         [0, 1, 1, 1, 5, 5, 5],
         [0, 1, 1, 1, 6, 6, 6],
         [0, 1, 1, 1, 7, 7, 7, 6]]
    return y[ndim]


def get_y_rosenbrock(ndim):
    y = [[409, 4, 1, 0, 401],
         [810, 4, 2, 400, 4002],
         [3, 0, 1212, 804, 2705, 17913, 24330],
         [4, 0, 1616, 808, 14814, 30038, 68450],
         [5, 0, 2020, 808, 50930, 126154, 164579],
         [6, 0, 2424, 1208, 135055, 210303, 349519, 51031]]
    return y[ndim]


def get_y_schwefel12(ndim):
    y = [[4, 1, 0, 5, 20],
         [8, 2, 0, 6, 24],
         [0, 30, 30, 2, 146, 10, 18],
         [0, 55, 55, 3, 371, 19, 55],
         [0, 91, 91, 7, 812, 28, 195],
         [0, 140, 140, 8, 1596, 44, 564, 812]]
    return y[ndim]


def get_y_griewank(ndim):
    y = [[1.066895, 0.589738, 0, 0.589738, 1.066895],
         [1.029230, 0.656567, 0, 0.656567, 1.029230],
         [0, 0.698951, 0.698951, 0.698951, 1.001870, 1.001870, 0.886208],
         [0, 0.728906, 0.728906, 0.728906, 1.017225, 1.017225, 0.992641],
         [0, 0.751538, 0.751538, 0.751538, 1.020074, 1.020074, 0.998490],
         [0, 0.769431, 0.769431, 0.769431, 1.037353, 1.037353, 1.054868, 1.024118]]
    return y[ndim]


def get_y_ackley(ndim):
    y = [[6.593599, 3.625384, 0, 3.625384, 6.593599],
         [6.593599, 3.625384, 0, 3.625384, 6.593599],
         [0, 3.625384, 3.625384, 3.625384, 8.434694, 8.434694, 8.434694],
         [0, 3.625384, 3.625384, 3.625384, 9.697286, 9.697286, 9.697286],
         [0, 3.625384, 3.625384, 3.625384, 10.821680, 10.821680, 10.821680],
         [0, 3.625384, 3.625384, 3.625384, 11.823165, 11.823165, 11.823165, 10.275757]]
    return y[ndim]


def get_y_rastrigin(ndim):
    y = [[8, 2, 0, 2, 8],
         [12, 3, 0, 3, 12],
         [0, 4, 4, 4, 30, 30, 30],
         [0, 5, 5, 5, 55, 55, 55],
         [0, 6, 6, 6, 91, 91, 91],
         [0, 7, 7, 7, 140, 140, 140, 91]]
    return y[ndim]
