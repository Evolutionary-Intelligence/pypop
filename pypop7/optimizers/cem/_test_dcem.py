import unittest
import time
import numpy as np

from pypop7.benchmarks.base_functions import sphere, cigar
from pypop7.optimizers.cem.dcem import DCEM as Solver


class TestSECEM(unittest.TestCase):
    def test_optimize(self):
        start_run = time.time()
        ndim_problem = 10
        for f in [sphere, cigar]:
            print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
            problem = {'fitness_function': f,
                       'ndim_problem': ndim_problem,
                       'lower_boundary': -5 * np.ones((ndim_problem,)),
                       'upper_boundary': 5 * np.ones((ndim_problem,))}
            options = {'max_function_evaluations': 2e4,
                       'fitness_threshold': 1e-10,
                       'seed_rng': 0,
                       'x': 4 * np.ones((ndim_problem,)),  # mean
                       'n_parents': 10,
                       'n_individuals': 20,
                       'sigma': 1.0,
                       'lml_verbose': 0,
                       'lml_eps': 1e-3,
                       'verbose_frequency': 20,
                       'record_fitness': True,
                       'record_fitness_frequency': 1}
            solver = Solver(problem, options)
            results = solver.optimize()
            print(results)
            print('*** Runtime: {:7.5e}'.format(time.time() - start_run))


if __name__ == '__main__':
    unittest.main()
