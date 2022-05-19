"""Repeat Fig. 3 (a) and (c) from the following paper:
    He, X., Zheng, Z. and Zhou, Y., 2021.
    MMES: Mixture model-based evolution strategy for large-scale optimization.
    IEEE Transactions on Evolutionary Computation, 25(2), pp.320-333.
    https://ieeexplore.ieee.org/abstract/document/9244595
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from optimizers.es.mmes import MMES

sns.set_theme(style='darkgrid')


if __name__ == '__main__':
    # plot Fig. 3 (a) ellipsoid and (c) rosenbrock
    from benchmarks.base_functions import ellipsoid, rosenbrock
    ndim_problem = 1000
    for f in [ellipsoid, rosenbrock]:
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -5 * np.ones((ndim_problem,)),
                   'upper_boundary': 5 * np.ones((ndim_problem,))}
        options = {'fitness_threshold': 1e-8,
                   'max_function_evaluations': 1e8,
                   'seed_rng': 2022,  # not given in the original paper
                   'sigma': 3,
                   'verbose_frequency': 20000,
                   'record_fitness': True,
                   'record_fitness_frequency': 1}
        mmes = MMES(problem, options)
        results = mmes.optimize()
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.yticks(ticks=[1e-5, 1e0, 1e5])
        plt.xticks(ticks=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
        plt.xlim([1e2, 1e8])
        fitness = results['fitness']
        plt.plot(fitness[:, 0], fitness[:, 1], 'r-')
        plt.xlabel('FEs')
        plt.ylabel('Objective Value')
        plt.show()
