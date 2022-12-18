"""Repeat the following paper for `ESA`:
    Siarry, P., Berthiau, G., Durdin, F. and Haussy, J., 1997.
    Enhanced simulated annealing for globally minimizing functions of many-continuous variables.
    ACM Transactions on Mathematical Software, 23(2), pp.209-228.
    https://dl.acm.org/doi/abs/10.1145/264029.264043

    Since several implementation details were not very clear in the original paper, it is hard to
      repeat the original experiments *perfectly*, according to our experiences.
    Our current implementation could generate the *relatively close but not perfect* result on the following
      test function, depending on your level of acceptance.
    We still expect to improve it, in order to match the original paper as close as possible in the future
      (if possible).

    The fuzzy details are presented below, which can cause different implementations with different performances:
    1. Initial FOBJ variation average, DGYINI, obtained from (typically) 50 uphill moves performed
      before beginning SA minimization itself.
    2. Step 2: Space Partitioning.
    3. Those elementary operations on only p coordinates move the point XSTART to the point XTRY
      satisfying all hyperrectangular box constraints.
    4. MSOTST = 0 (this is a typo in the original paper).
    5. Other systematic experiments are still in progress in order to obtain, if possible, a rule for
      the p-optimal choice, adapted to FOBJ-specific characteristics.
"""
import numpy as np

from pypop7.benchmarks.base_functions import rosenbrock
from pypop7.optimizers.sa.esa import ESA


if __name__ == '__main__':
    problem = {'fitness_function': rosenbrock,
               'ndim_problem': 50,
               'upper_boundary': 10 * np.ones((50,)),
               'lower_boundary': -5 * np.ones((50,))}
    options = {'max_function_evaluations': 78224,
               'seed_rng': 1,  # undefined in the original paper
               'p': 1,
               'saving_fitness': 1000,
               'verbose': 10000}
    esa = ESA(problem, options)
    results = esa.optimize()
    # FOBJ initial-value average: 6472239 (from 50 samples) vs 5e5 (from the original paper)
    # our implementation starts from a worse starting point than the original paper.
    print(results)  # 49.13408124446869 vs 8.8 (from the original paper)
