import unittest
import time
import numpy as np

from pypop7.benchmarks.base_functions import sphere, cigar, rosenbrock
from pypop7.optimizers.ds.powell import POWELL as Solver

class TestHJDSM(unittest.TestCase):
    def test_optimize(self):
        start_run = time.time()
        ndim_problem = 10
        for f in [sphere, cigar, rosenbrock]:
            print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
            problem = {'fitness_function': f,
                       'ndim_problem': ndim_problem,
                       'lower_boundary': -5 * np.ones((ndim_problem,)),
                       'upper_boundary': 5 * np.ones((ndim_problem,))}
            options = {'max_function_evaluations': 2e4,
                       'fitness_threshold': 1e-8,
                       'seed_rng': 0,
                       'verbose_frequency': 1,
                       'record_fitness': True,
                       'record_fitness_frequency': 1}
            solver = Solver(problem, options)
            results = solver.optimize()
            print(results)
            print('*** Runtime: {:7.5e}'.format(time.time() - start_run))


if __name__ == '__main__':
    unittest.main()
