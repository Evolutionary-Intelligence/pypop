"""Repeat the following paper for `CSA`:
    Corana, A., Marchesi, M., Martini, C. and Ridella, S., 1987.
    Minimizing multimodal functions of continuous variables with the "simulated annealing" algorithm.
    ACM Transactions on Mathematical Software, 13(3), pp.262-280.
    https://dl.acm.org/doi/abs/10.1145/29380.29864
    https://dl.acm.org/doi/10.1145/66888.356281

    Luckily our Python code could repeat the data reported in the original paper *well* on at least the
    following test functions, although the settings of some hyper-parameters are not given clearly in
    the original paper. Therefore, we argue that its repeatability could be **well-documented** on at
    least these test functions.
"""
import numpy as np

from pypop7.benchmarks.base_functions import rosenbrock
from pypop7.optimizers.sa.csa import CSA


if __name__ == '__main__':
    problem = {'fitness_function': rosenbrock,
               'ndim_problem': 2,
               'upper_boundary': 2000*np.ones((2,)),
               'lower_boundary': -2000*np.ones((2,))}
    options = {'max_function_evaluations': 512001,
               'seed_rng': 0,  # undefined in the original paper
               'sigma': 3.0,  # undefined in the original paper
               'temperature': 50000,
               'x': np.array([1, 1443]),
               'verbose': 1000,
               'saving_fitness': 1000}
    csa = CSA(problem, options)
    results = csa.optimize()
    print(results)  # 0.000000e+00 vs 1.6e-9 (from the original paper)

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': 2,
               'upper_boundary': 2000*np.ones((2,)),
               'lower_boundary': -2000*np.ones((2,))}
    options = {'max_function_evaluations': 488001,
               'seed_rng': 0,  # undefined in the original paper
               'sigma': 3.0,  # undefined in the original paper
               'temperature': 50000,
               'x': np.array([1.2, 1]),
               'verbose': 1000,
               'saving_fitness': 1000}
    csa = CSA(problem, options)
    results = csa.optimize()
    print(results)  # 6.33127974e-09 vs 2e-8 (from the original paper)

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': 4,
               'upper_boundary': 200*np.ones((4,)),
               'lower_boundary': -200*np.ones((4,))}
    options = {'max_function_evaluations': 1288001,
               'seed_rng': 0,  # undefined in the original paper
               'sigma': 2.0,  # undefined in the original paper
               'temperature': 1e7,
               'x': 101*np.ones((4,)),
               'verbose': 1000,
               'saving_fitness': 1000}
    csa = CSA(problem, options)
    results = csa.optimize()
    print(results)  # 1.82666130e-07 vs 5e-7 (from the original paper)

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': 4,
               'upper_boundary': 200*np.ones((4,)),
               'lower_boundary': -200*np.ones((4,))}
    options = {'max_function_evaluations': 1296001,
               'seed_rng': 0,  # undefined in the original paper
               'sigma': 1.5,  # undefined in the original paper
               'temperature': 1e7,
               'x': -99*np.ones((4,)),
               'verbose': 1000,
               'saving_fitness': 1000}
    csa = CSA(problem, options)
    results = csa.optimize()
    print(results)  # 1.30552141e-07 vs 7.4e-8 (from the original paper)
