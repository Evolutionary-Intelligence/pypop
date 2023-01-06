"""Repeat the following paper for `SAMAES`:
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere
from pypop7.optimizers.es.samaes import SAMAES


if __name__ == '__main__':
    problem = {'fitness_function': sphere,
               'ndim_problem': 30}
    options = {'seed_rng': 1,  # undefined in the original paper
               'fitness_threshold': 1.3899e-18,
               'n_individuals': 12,
               'n_parents': 3,
               'saving_fitness': 1,
               'x': np.ones((30,)),
               'sigma': 1,
               'is_restart': False}
    samaes = SAMAES(problem, options)
    results = samaes.optimize()
    print(results)
    print(results['best_so_far_y'])  # 1.38004278656276e-18
    print(results['sigma'])  # 1.387885288290368e-10
    print(results['_n_generations'])  # 473
