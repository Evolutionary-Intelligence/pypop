"""Repeat the following paper for `LMCMAES`:
    Loshchilov, I., 2014, July.
    A computationally efficient limited memory CMA-ES for large scale optimization.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 397-404). ACM.
    https://dl.acm.org/doi/abs/10.1145/2576768.2598294 (See Algorithm 6 for details.)
    https://sites.google.com/site/lmcmaeses/ (Note that use the modified C++ (rather Matlab) version.)

    Luckily our code could repeat the data reported in the original paper *well*.
    Therefore, we argue that the repeatability of `LMCMAES` could be **well-documented**.

    For the C++ code, you may first need to change Line 1594 `void main()` to `int main()` and
    Line 1616 `bool sample_symmetry=false` to `bool sample_symmetry=true`.
    Then you can run the following command successfully (if using Windows OS).
    $ gcc lmcma.cpp -lstdc++ -o lmcma.exe
    $ .\lmcma.exe
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import ellipsoid, discus, cigar
from pypop7.optimizers.es.lmcmaes import LMCMAES as Solver


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 2 * 128
    for f in [ellipsoid, discus, cigar]:
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -5 * np.ones((ndim_problem,)),
                   'upper_boundary': 5 * np.ones((ndim_problem,))}
        options = {'max_function_evaluations': 1e8,
                   'fitness_threshold': 1e-10,
                   'seed_rng': 0,  # undefined in the original paper
                   'sigma': 5,
                   'verbose': 2000,
                   'saving_fitness': 200000}
        solver = Solver(problem, options)
        results = solver.optimize()
        print(results)
        print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
    # compare `n_function_evaluations`:
    #   ellipsoid: 4.780563e+06 vs 4893900 (from the C++ code)
    #   discus: 1.175605e+06 vs 1216460 (from the C++ code)
    #   cigar:  5.4790e+04 vs 60520 (from the C++ code)
