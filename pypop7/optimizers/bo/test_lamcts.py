import unittest
import time
import numpy as np

from pypop7.benchmarks.base_functions import ackley
from pypop7.optimizers.bo.lamcts import LAMCTS as Solver


class TestLAMCTS(unittest.TestCase):
    def test_optimize(self):
        start_run = time.time()
        ndim_problem = 20
        for f in [ackley]:
            print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
            problem = {'fitness_function': f,
                       'ndim_problem': ndim_problem,
                       'lower_boundary': -5 * np.ones((ndim_problem,)),
                       'upper_boundary': 10 * np.ones((ndim_problem,))}
            options = {'max_function_evaluations': 1e2,
                       'fitness_threshold': 1e-5,
                       'seed_rng': 0,
                       'n_individuals': 40,  # the number of random samples used in initialization
                       'Cp': 1,  # Cp for MCTS
                       'leaf_size': 10,  # tree leaf size
                       'kernel_type': "rbf",  # used in SVM
                       'gamma_type': "auto",  # used in SVM
                       'solver_type': 'bo',  # solver can be bo or turbo
                       'verbose_frequency': 1,
                       'record_fitness': False,
                       'record_fitness_frequency': 1}
            solver = Solver(problem, options)
            results = solver.optimize()
            print(results)
            print('*** Runtime: {:7.5e}'.format(time.time() - start_run))


if __name__ == '__main__':
    unittest.main()
