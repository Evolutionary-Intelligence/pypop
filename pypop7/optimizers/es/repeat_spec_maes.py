"""Repeat Table 2 and 3 from the following paper:
    Ros, R. and Hansen, N., 2008, September.
    A simple modification in CMA-ES achieving linear time and space complexity.
    In International Conference on Parallel Problem Solving from Nature (pp. 296-305).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-540-87700-4_30
"""
import numpy as np

from benchmarks.base_functions import rosenbrock, ellipsoid
from optimizers.es.sepcmaes import SEPCMAES
from benchmarks.base_functions import _squeeze_and_check


def different_powers(x):
    x = np.abs(_squeeze_and_check(x, True))
    y = np.sum(np.power(x, 2 + 10 * np.linspace(0, 1, x.size)))
    return y


def hyper_ellipsoid(x):
    x, y = _squeeze_and_check(x), 0
    for i in range(x.size):
        y += np.power(x[i] * (i + 1), 2)
    return y


if __name__ == '__main__':
    # Repeat Table 2
    print("Table 2")
    ndim_problem = 30
    for f in [hyper_ellipsoid, different_powers, rosenbrock]:
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem}
        options = {'max_function_evaluations': 200 * 1e3,
                   'fitness_threshold': 1e-6,
                   'seed_rng': 2022,  # not given in the original paper
                   'verbose_frequency': 1000,
                   'record_fitness': True,
                   'record_fitness_frequency': 1000}
        if f == hyper_ellipsoid:
            options['fitness_threshold'] = 1e-10
            options['sigma'] = 1
            options['x'] = np.ones((ndim_problem,))
        elif f == different_powers:
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
        print(f.__name__, "dim = ", problem['ndim_problem'], " fitness threshold = ",
              options['fitness_threshold'])
        print("functions evaluations = ", results['n_function_evaluations'])

    # Repeat Table 3
    print("Table 3")
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
                   'stagnation': np.Inf}
        if f == ellipsoid:
            options['sigma'] = 1
            options['x'] = np.ones((ndim_problem,))
        elif f == rosenbrock:
            options['sigma'] = 0.1
            options['x'] = np.zeros((ndim_problem,))
        solver = SEPCMAES(problem, options)
        results = solver.optimize()
        print(results)
        print(f.__name__, "dim = ", problem['ndim_problem'], " fitness threshold = ",
              options['fitness_threshold'])
        print("function evaluations = ", results['n_function_evaluations'])
