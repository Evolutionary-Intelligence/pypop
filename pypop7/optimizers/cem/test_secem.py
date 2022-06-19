import unittest
import time
import numpy as np

from pypop7.benchmarks.base_functions import sphere, cigar
from pypop7.optimizers.cem.secem import SECEM as Solver

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
            options = {'max_function_evaluations': 2e6,
                       'fitness_threshold': 1e-10,
                       'seed_rng': 0,
                       'x': 4 * np.ones((ndim_problem,)),  # mean
                       'n_parents': 20,
                       'n_individuals': 150,
                       'fraction_reused_elites': 0.3,
                       'sigma': 0.5,
                       'noise_beta': 2.0,
                       'verbose_frequency': 200,
                       'record_fitness': True,
                       'record_fitness_frequency': 1}
            solver = Solver(problem, options)
            results = solver.optimize()
            print(results)
            print('*** Runtime: {:7.5e}'.format(time.time() - start_run))


if __name__ == '__main__':
    unittest.main()
