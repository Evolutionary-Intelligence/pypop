"""Repeat the following paper for `FCMAES`:
    Li, Z., Zhang, Q., Lin, X. and Zhen, H.L., 2018.
    Fast covariance matrix adaptation for large-scale black-box optimization.
    IEEE Transactions on Cybernetics, 50(5), pp.2073-2083.
    https://ieeexplore.ieee.org/abstract/document/8533604

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/fcmaes

    Luckily our Python code could repeat the data reported in the paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import cigar, ellipsoid, discus, different_powers, rosenbrock
from pypop7.optimizers.es.fcmaes import FCMAES


def plot(function, ndim):
    plt.figure()
    result = pickle.load(open(function + '_' + str(ndim) + '.pickle', 'rb'))
    plt.yscale('log')
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='r')
    plt.ylim([1e-10, 1e10])
    plt.yticks(ticks=[1e-10, 1e-5, 1e0, 1e5, 1e10])
    if function == 'cigar':
        plt.xlim([0e5, 5e5])
        plt.xticks([0e5, 1e5, 2e5, 3e5, 4e5, 5e5])
    elif function == 'ellipsoid':
        plt.xlim([0e7, 2.5e7])
        plt.xticks([0e7, 0.5e7, 1e7, 1.5e7, 2e7, 2.5e7])
    elif function == 'discus':
        plt.xlim([0e7, 5e7])
        plt.xticks([0e7, 1e7, 2e7, 3e7, 4e7, 5e7])
    elif function == 'different_powers':
        plt.xlim([0e6, 10e6])
        plt.xticks([0e6, 2e6, 4e6, 6e6, 8e6, 10e6])
    elif function == 'rosenbrock':
        plt.xscale('log')
        plt.xlim([1e2, 1e8])
        plt.xticks([1e2, 1e4, 1e6, 1e8])
    plt.xlabel("evaluations")
    plt.ylabel("objective value")
    plt.title(function.capitalize())
    plt.show()


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    ndim_problem = 1024
    for f in [rosenbrock, cigar, ellipsoid, discus, different_powers]:
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -10 * np.ones((ndim_problem,)),
                   'upper_boundary': 10 * np.ones((ndim_problem,))}
        options = {'fitness_threshold': 1e-8,
                   'seed_rng': 1,  # not given in the original paper
                   'sigma': 20.0 / 3.0,
                   'is_restart': False,
                   'saving_fitness': 1}
        rmes = FCMAES(problem, options)
        results = rmes.optimize()
        print(results)
        pickle.dump(results, open(f.__name__ + '_' + str(ndim_problem) + '.pickle', 'wb'))
        plot(f.__name__, ndim_problem)
