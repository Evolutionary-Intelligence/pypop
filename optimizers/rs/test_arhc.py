import unittest
import numpy as np

from benchmarks.base_functions import ellipsoid
from optimizers.rs.arhc import ARHC


class TestARHC(unittest.TestCase):
    def test_run(self):
        ndim_problem = 1000
        problem = {'fitness_function': ellipsoid,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': np.zeros((ndim_problem,)),
                   'upper_boundary': 2 * np.ones((ndim_problem,))}
        options = {'max_function_evaluations': 1e6,
                   'fitness_threshold': 1e-10,
                   'seed_rng': 0,
                   'x': np.ones((ndim_problem,)),
                   'record_options': {'record_fitness': True,
                                      'frequency_record_fitness': 2000}}

        print('*' * 7 + ' Run 1 ' + '*' * 7)
        arhc = ARHC(problem, options)
        results = arhc.optimize()
        print(results)

        print('*' * 7 + ' Run 2 ' + '*' * 7)
        del options['x']
        arhc = ARHC(problem, options)
        results = arhc.optimize()
        print(results)


if __name__ == '__main__':
    unittest.main()
