"""Repeat Fig. 3(d) from the following paper:
    Li, Z. and Zhang, Q., 2018.
    A simple yet efficient evolution strategy for large-scale black-box optimization.
    IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
    https://ieeexplore.ieee.org/abstract/document/8080257
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from optimizers.es.rmes import RMES

sns.set_theme(style='darkgrid')


if __name__ == '__main__':
    # plot Fig. 3(d)
    from benchmarks.base_functions import rosenbrock
    ndim_problem = 1000
    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -10 * np.ones((ndim_problem,)),
               'upper_boundary': 10 * np.ones((ndim_problem,))}
    options = {'fitness_threshold': 1e-8,
               'max_function_evaluations': 1e8,
               'seed_rng': 2022,  # not given in the original paper
               'sigma': 20 / 3,
               'verbose_frequency': 20000,
               'record_fitness': True,
               'record_fitness_frequency': 1}
    rmes = RMES(problem, options)
    results = rmes.optimize()
    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.yticks(ticks=[1e-8, 1e-4, 1e0, 1e4, 1e8])
    plt.xticks(ticks=[1e2, 1e4, 1e6, 1e8])
    plt.xlim([1e2, 1e8])
    fitness = results['fitness']
    plt.plot(fitness[:, 0], fitness[:, 1], 'r-')
    plt.xlabel('Function Evaluations')
    plt.ylabel('Objective Value')
    plt.show()
