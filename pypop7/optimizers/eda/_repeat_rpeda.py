import numpy as np

from pypop7.benchmarks.base_functions import rosenbrock
from pypop7.optimizers.eda.rpeda import RPEDA


if __name__ == '__main__':
    ndim_problem = 1000
    for f in [rosenbrock]:
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -100 * np.ones((ndim_problem,)),
                   'upper_boundary': 100 * np.ones((ndim_problem,))}
        options = {'max_function_evaluations': 6e5,
                   'm': 1000,
                   'seed_rng': 0,
                   'verbose': 100,
                   'saving_fitness': 1}
        rpeda = RPEDA(problem, options)
        results = rpeda.optimize()
        print(f.__name__, results['best_so_far_y'])
        # rosenbrock 987.6557605512518 vs mean 1614.7 std 249.44 (from the original paper)
