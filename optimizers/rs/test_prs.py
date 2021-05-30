import unittest
import numpy as np
import time

from benchmarks.base_functions import ellipsoid, rosenbrock, rastrigin
from optimizers.rs.prs import PRS


class TestPRS(unittest.TestCase):
    def test_run(self):
        start_run = time.time()
        ndim_problem = 1000
        for f in [ellipsoid, rosenbrock, rastrigin]:
            print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
            problem = {'fitness_function': f,
                       'ndim_problem': ndim_problem,
                       'lower_boundary': -5 * np.zeros((ndim_problem,)),
                       'upper_boundary': 5 * np.ones((ndim_problem,))}
            options = {'max_function_evaluations': 2e6,
                       'fitness_threshold': 1e-10,
                       'seed_rng': 0,
                       'x': 4 * np.ones((ndim_problem,)),
                       'verbose_options': {'frequency_verbose': 200000},
                       'record_options': {'record_fitness': True,
                                          'frequency_record_fitness': 200000}}
            prs = PRS(problem, options)
            results = prs.optimize()
            print(results)
            print('*** Runtime: {:7.5e}'.format(time.time() - start_run))


if __name__ == '__main__':
    unittest.main()
