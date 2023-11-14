"""
Test the common optimizer interface.
"""
import unittest

import numpy as np

from pypop7.benchmarks.base_functions import rosenbrock
from pypop7.optimizers.core.optimizer import Terminations
from pypop7.optimizers.de.cde import CDE
from pypop7.optimizers.es.fmaes import FMAES


class TestOptimizer(unittest.TestCase):
    def test_early_stopping(self):
        ndim_problem = 10
        early_stopping_threshold = 1e-8
        early_stopping_evaluations = 10000

        # This function is needed to avoid the fitness_threshold condition
        rosenbrock_plus_one = lambda x: rosenbrock(x) + 1

        for Solver in [CDE, FMAES]:
            problem = {'fitness_function': rosenbrock_plus_one,
                       'ndim_problem': ndim_problem,
                       'lower_boundary': -5 * np.ones((ndim_problem,)),
                       'upper_boundary': 5 * np.ones((ndim_problem,))}
            options = {'max_function_evaluations': 2e6,
                       'fitness_threshold': 1e-10,
                       'seed_rng': 0,
                       'early_stopping_threshold': early_stopping_threshold,
                       'early_stopping_evaluations': early_stopping_evaluations,
                       'verbose': 200,
                       'sigma': 1e-3,
                       }
            solver = Solver(problem, options)
            results = solver.optimize()
            self.assertTrue(results['termination_signal'], Terminations.EARLY_STOPPING)


if __name__ == '__main__':
    unittest.main()
