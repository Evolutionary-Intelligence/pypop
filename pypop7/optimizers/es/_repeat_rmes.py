"""Repeat the following paper for `RMES`:
    Li, Z. and Zhang, Q., 2018.
    A simple yet efficient evolution strategy for large-scale black-box optimization.
    IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
    https://ieeexplore.ieee.org/abstract/document/8080257

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/rmes

    Luckily our code could repeat the data reported in the original paper *well*.
    Therefore, we argue that the repeatability of `RMES` could be **well-documented**.
"""
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import ellipsoid, discus, rosenbrock
from pypop7.optimizers.es.rmes import RMES


def plot(function, ndim):
    plt.figure()
    result = pickle.load(open(function + '_' + str(ndim) + '.pickle', 'rb'))
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='orange')
    plt.xlim([1e2, 1e8])
    plt.xticks([1e2, 1e4, 1e6, 1e8])
    if function == 'ellipsoid':
        plt.ylim([1e-8, 1e10])
        plt.yticks(ticks=[1e-8, 1e-4, 1e0, 1e4, 1e8])
    elif function == 'discus':
        plt.ylim([1e-8, 1e6])
        plt.yticks(ticks=[1e-8, 1e-4, 1e0, 1e4])
    elif function == 'rosenbrock':
        plt.ylim([1e-8, 1e10])
        plt.yticks(ticks=[1e-8, 1e-4, 1e0, 1e4, 1e8])
    plt.xlabel("Function Evaluations")
    plt.ylabel("Objective Value")
    plt.title(function.capitalize())
    plt.show()


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    ndim_problem = 1000
    for f in [ellipsoid, discus, rosenbrock]:
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -10 * np.ones((ndim_problem,)),
                   'upper_boundary': 10 * np.ones((ndim_problem,))}
        options = {'fitness_threshold': 1e-8,
                   'max_function_evaluations': 1e8,
                   'seed_rng': 2022,  # not given in the original paper
                   'sigma': 20 / 3,
                   'verbose': 20000,
                   'saving_fitness': 1}
        rmes = RMES(problem, options)
        results = rmes.optimize()
        pickle.dump(results, open(f.__name__ + '_' + str(ndim_problem) + '.pickle', 'wb'))
        plot(f.__name__, ndim_problem)
