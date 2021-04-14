import unittest
import numpy as np

from benchmarks.base_functions import ellipsoid
from benchmarks.shifted_functions import generate_shift_vector
from benchmarks.shifted_functions import ellipsoid as shifted_ellipsoid
from benchmarks.rotated_functions import generate_rotation_matrix
from benchmarks.continuous_functions import ellipsoid as rotated_shifted_ellipsoid
from optimizers.pso.clpso import CLPSO


class TestCLPSO(unittest.TestCase):
    def test_run(self):
        ndim_problem = 1000
        options = {'max_function_evaluations': 1e6,
                   'fitness_threshold': 1e-10,
                   'seed_rng': 0,
                   'record_options': {'record_fitness': True,
                                      'frequency_record_fitness': 2000}}

        print('*' * 7 + ' Run 1 ' + '*' * 7)
        problem = {'fitness_function': ellipsoid,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': np.zeros((ndim_problem,)),
                   'upper_boundary': 2 * np.ones((ndim_problem,))}
        clpso = CLPSO(problem, options)
        results = clpso.optimize()
        print(results)

        print('*' * 7 + ' Run 2 ' + '*' * 7)
        generate_shift_vector(ellipsoid, ndim_problem, 1.2, 1.7, 2021)
        problem = {'fitness_function': shifted_ellipsoid,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': np.zeros((ndim_problem,)),
                   'upper_boundary': 2 * np.ones((ndim_problem,))}
        clpso = CLPSO(problem, options)
        results = clpso.optimize()
        print(results)

        print('*' * 7 + ' Run 3 ' + '*' * 7)
        generate_rotation_matrix(ellipsoid, ndim_problem, 2022)
        problem = {'fitness_function': rotated_shifted_ellipsoid,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': np.zeros((ndim_problem,)),
                   'upper_boundary': 2 * np.ones((ndim_problem,))}
        clpso = CLPSO(problem, options)
        results = clpso.optimize()
        print(results)


if __name__ == '__main__':
    unittest.main()
