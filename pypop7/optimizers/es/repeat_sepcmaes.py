"""Repeat Table 2 and 3 from the following paper:
    Ros, R. and Hansen, N., 2008, September.
    A simple modification in CMA-ES achieving linear time and space complexity.
    In International Conference on Parallel Problem Solving from Nature (pp. 296-305).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-540-87700-4_30
"""
import numpy as np

from pypop7.benchmarks.base_functions import rosenbrock, ellipsoid
from pypop7.optimizers.es.sepcmaes import SEPCMAES


def _different_powers(x):
    y = np.sum(np.power(np.abs(x), 2 + 10 * np.linspace(0, 1, x.size)))
    return y


def _hyper_ellipsoid(x):
    y = 0
    for i in range(x.size):
        y += np.power(x[i] * (i + 1), 2)
    return y


if __name__ == '__main__':
    # Table 2
    ndim_problem = 30
    for f in [_hyper_ellipsoid, _different_powers, rosenbrock]:
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem}
        options = {'max_function_evaluations': 200 * 1e3,
                   'fitness_threshold': 1e-6,
                   'seed_rng': 2022,  # not given in the original paper
                   'verbose_frequency': 1000,
                   'record_fitness': True,
                   'record_fitness_frequency': 1000,
                   'is_restart': False}
        if f == _hyper_ellipsoid:
            options['fitness_threshold'] = 1e-10
            options['sigma'] = 1
            options['x'] = np.ones((ndim_problem,))
        elif f == _different_powers:
            options['fitness_threshold'] = 1e-20
            options['sigma'] = 1
            options['x'] = np.ones((ndim_problem,))
        elif f == rosenbrock:
            options['fitness_threshold'] = 1e-6
            options['sigma'] = 0.1
            options['x'] = np.zeros((ndim_problem,))
        solver = SEPCMAES(problem, options)
        results = solver.optimize()
        print(results)
        print(f.__name__, 'dim = ', problem['ndim_problem'],
              ' fitness threshold = ', options['fitness_threshold'])
        print('functions evaluations = ', results['n_function_evaluations'])
    # Table 3
    ndim_problem = 20
    for f in [ellipsoid, rosenbrock]:
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem}
        options = {'max_function_evaluations': 200 * 1e3,
                   'fitness_threshold': 1e-9,
                   'seed_rng': 2022,  # not given in the original paper
                   'verbose_frequency': 1000,
                   'record_fitness': True,
                   'record_fitness_frequency': 1000,
                   'is_restart': False}
        if f == ellipsoid:
            options['sigma'] = 1
            options['x'] = np.ones((ndim_problem,))
        elif f == rosenbrock:
            options['sigma'] = 0.1
            options['x'] = np.zeros((ndim_problem,))
        solver = SEPCMAES(problem, options)
        results = solver.optimize()
        print(results)
        print(f.__name__, 'dim = ', problem['ndim_problem'],
              ' fitness threshold = ', options['fitness_threshold'])
        print('function evaluations = ', results['n_function_evaluations'])
