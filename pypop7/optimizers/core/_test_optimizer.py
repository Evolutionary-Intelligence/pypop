import unittest

import numpy as np

from pypop7.benchmarks.base_functions import rosenbrock
from pypop7.optimizers.core.optimizer import Terminations
from pypop7.optimizers.de.cde import CDE
from pypop7.optimizers.es.fmaes import FMAES


class TestOptimizer(unittest.TestCase):
    def test_early_stopping_1(self):
        ndim_problem = 10
        problem = {'fitness_function': rosenbrock,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -5.0*np.ones((ndim_problem,)),
                   'upper_boundary': 5.0*np.ones((ndim_problem,))}
        options = {'seed_rng': 0,
                   'early_stopping_threshold': 1e-7,
                   'early_stopping_evaluations': 10000}
        solver = CDE(problem, options)
        results = solver.optimize()
        self.assertTrue(results['termination_signal'], Terminations.EARLY_STOPPING)
        # * Generation 840: best_so_far_y 5.02011e-08, min(y) 5.02011e-08 & Evaluations 84100
        # * Generation 935: best_so_far_y 1.24183e-10, min(y) 1.24183e-10 & Evaluations 93531

    def test_early_stopping_2(self):
        ndim_problem = 10
        problem = {'fitness_function': rosenbrock,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -5.0*np.ones((ndim_problem,)),
                   'upper_boundary': 5.0*np.ones((ndim_problem,))}
        options = {'seed_rng': 0,
                   'sigma': 1e-3,
                   'early_stopping_threshold': 1e-7,
                   'early_stopping_evaluations': 10000}
        solver = FMAES(problem, options)
        results = solver.optimize()
        self.assertTrue(results['termination_signal'], Terminations.EARLY_STOPPING)
        # * Generation 810: best_so_far_y 2.37215e-08, min(y) 3.44843e-08 & Evaluations 8110
        # * Generation 443: best_so_far_y 2.70942e-15, min(y) 3.98658e+00 & Evaluations 18033


if __name__ == '__main__':
    unittest.main()
