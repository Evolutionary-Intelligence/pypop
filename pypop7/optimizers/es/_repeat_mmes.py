"""Repeat the following paper for `MMES`:
    He, X., Zheng, Z. and Zhou, Y., 2021.
    MMES: Mixture model-based evolution strategy for large-scale optimization.
    IEEE Transactions on Evolutionary Computation, 25(2), pp.320-333.
    https://ieeexplore.ieee.org/abstract/document/9244595

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/mmes

    Luckily our Python code could repeat the data reported in the paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import discus, cigar, ellipsoid, rosenbrock
from pypop7.optimizers.es.mmes import MMES as Solver


def plot(function, ndim):
    plt.figure()
    result = pickle.load(open(function + '_' + str(ndim) + '.pickle', 'rb'))
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='r')
    if function == 'discus':
        plt.xlim([1e2, 1e8])
        plt.ylim([1e-8, 1e7])
        plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
        plt.yticks(ticks=[1e-5, 1e0, 1e5])
    elif function == 'cigar':
        plt.xlim([1e2, 1e7])
        plt.ylim([1e-8, 1e10])
        plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6])
        plt.yticks(ticks=[1e-5, 1e0, 1e5])
    elif function == 'ellipsoid':
        plt.xlim([1e2, 1e8])
        plt.ylim([1e-8, 1e9])
        plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
        plt.yticks(ticks=[1e-5, 1e0, 1e5])
    elif function == 'rosenbrock':
        plt.xlim([1e2, 1e8])
        plt.ylim([1e-8, 1e8])
        plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
        plt.yticks(ticks=[1e-5, 1e0, 1e5])
    plt.xlabel("FEs")
    plt.ylabel("Objective Value")
    plt.title(function.capitalize())
    plt.show()


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    ndim_problem = 1000
    for f in [discus, cigar, ellipsoid, rosenbrock]:
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -5 * np.ones((ndim_problem,)),
                   'upper_boundary': 5 * np.ones((ndim_problem,))}
        options = {'fitness_threshold': 1e-8,
                   'max_function_evaluations': 1e8,
                   'seed_rng': 2022,  # not given in the original paper
                   'sigma': 3,
                   'saving_fitness': 1,
                   'is_restart': False}
        solver = Solver(problem, options)
        results = solver.optimize()
        pickle.dump(results, open(f.__name__ + '_' + str(ndim_problem) + '.pickle', 'wb'))
        plot(f.__name__, ndim_problem)
