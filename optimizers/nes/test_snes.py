import unittest

import numpy as np
from benchmarks.base_functions import ellipsoid
from snes import SNES


class TestSNES(unittest.TestCase):
    def test_run(self):
        ndim_problem = 1000
        problem = {'fitness_function': ellipsoid,
                   'ndim_problem': ndim_problem}
        options = {'max_function_evaluations': 1e6,
                   'fitness_threshold': 1e-10,
                   'seed_rng': 0,
                   'mu': np.ones((ndim_problem,))}
        snes = SNES(problem, options)
        results = snes.optimize()
        print(results)


if __name__ == '__main__':
    unittest.main()
