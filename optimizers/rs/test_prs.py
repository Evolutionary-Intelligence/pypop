import unittest
import numpy as np

from benchmarks.base_functions import ellipsoid
from optimizers.rs.prs import PRS


class TestPRS(unittest.TestCase):
    def test_run(self):
        ndim_problem = 1000
        problem = {'fitness_function': ellipsoid,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': np.zeros((ndim_problem,)),
                   'upper_boundary': 2 * np.ones((ndim_problem,))}
        options = {'max_function_evaluations': 1e6,
                   'fitness_threshold': 1e-10,
                   'seed_rng': 0,
                   'sampling_distribution': 0,
                   'x': np.ones((ndim_problem,)),
                   'record_options': {'record_fitness': True,
                                      'frequency_record_fitness': 2000}}

        print('*' * 7 + ' Run 1 ' + '*' * 7)
        prs = PRS(problem, options)
        print('sampling_distribution: ' + str(prs.sampling_distribution))
        results = prs.optimize()
        print(results)

        print('*' * 7 + ' Run 2 ' + '*' * 7)
        del options['x']
        prs = PRS(problem, options)
        print('sampling_distribution: ' + str(prs.sampling_distribution))
        results = prs.optimize()
        print(results)

        print('*' * 7 + ' Run 3 ' + '*' * 7)
        del options['sampling_distribution']
        prs = PRS(problem, options)
        print('sampling_distribution: ' + str(prs.sampling_distribution))
        results = prs.optimize()
        print(results)


if __name__ == '__main__':
    unittest.main()
