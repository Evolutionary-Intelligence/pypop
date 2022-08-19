import unittest
import time
import numpy as np

from pypop7.benchmarks.base_functions import schwefel222, sphere, schwefel12, step
from pypop7.optimizers.ep.cep import CEP as Solver


class TestSECEM(unittest.TestCase):
    def test_optimize(self):
        start_run = time.time()
        ndim_problem = 30
        for f in [step]:
            print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
            problem = {'fitness_function': f,
                       'ndim_problem': ndim_problem,
                       'lower_boundary': -100 * np.ones((ndim_problem,)),
                       'upper_boundary': 100 * np.ones((ndim_problem,))}
            options = {'max_function_evaluations': 2e6,
                       'fitness_threshold': 1e-10,
                       'seed_rng': 0,
                       'n_individuals': 100,
                       'q': 10,  # tournament size
                       'init_n': 3.0,
                       'verbose_frequency': 20,
                       'record_fitness': True,
                       'record_fitness_frequency': 1}
            solver = Solver(problem, options)
            results = solver.optimize()
            print(results)
            print('*** Runtime: {:7.5e}'.format(time.time() - start_run))


if __name__ == '__main__':
    unittest.main()
