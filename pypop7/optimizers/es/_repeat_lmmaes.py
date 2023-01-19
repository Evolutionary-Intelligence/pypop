"""Repeat the following paper for `LMMAES`:
    Loshchilov, I., Glasmachers, T. and Beyer, H.G., 2019.
    Large scale black-box optimization by limited-memory matrix adaptation.
    IEEE Transactions on Evolutionary Computation, 23(2), pp.353-358.
    https://ieeexplore.ieee.org/abstract/document/8410043

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/lmmaes

    Luckily our Python code could repeat the data reported in the original paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import sphere, cigar, ellipsoid, rosenbrock
from pypop7.optimizers.es.lmmaes import LMMAES as Solver


def plot(function):
    ndim = [128, 256, 512, 1024, 2048, 4096, 8192]
    colors = ['r', 'orange', 'y', 'limegreen', 'cyan', 'b', 'purple']
    plt.figure()
    for k in range(len(ndim)):
        result = pickle.load(open(function + '_' + str(ndim[k]) + '.pickle', 'rb'))
        plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color=colors[k],
                 label=str(ndim), linestyle='dashed')
    plt.yscale('log')
    plt.xscale('log')
    if function == 'sphere':
        plt.xlim([1e0, 1e6])
        plt.ylim([1e-10, 1e6])
        plt.xticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
        plt.yticks(ticks=[1e-10, 1e-5, 1e0, 1e5])
    if function == 'cigar':
        plt.xlim([1e2, 1e8])
        plt.ylim([1e-10, 1e11])
        plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
        plt.yticks(ticks=[1e-10, 1e-5, 1e0, 1e5, 1e10])
    if function == 'ellipsoid':
        plt.xlim([1e3, 1e9])
        plt.ylim([1e-10, 1e11])
        plt.xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
        plt.yticks(ticks=[1e-10, 1e-5, 1e0, 1e5, 1e10])
    if function == 'rosenbrock':
        plt.xlim([1e3, 1e9])
        plt.ylim([1e-10, 1e9])
        plt.xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
        plt.yticks(ticks=[1e-10, 1e-5, 1e0, 1e5, 1e10])
    plt.xlabel("number of function evaluations")
    plt.ylabel("objective")
    plt.title(function.capitalize())
    plt.show()


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    for f in [sphere, cigar, ellipsoid, rosenbrock]:
        for d in [128, 256, 512, 1024, 2048, 4096, 8192]:
            problem = {'fitness_function': f,
                       'ndim_problem': d}
            options = {'fitness_threshold': 1e-10,
                       'seed_rng': 0,  # undefined in the original paper
                       'x': 4 * np.ones((d,)),  # mean
                       'sigma': 3,
                       'verbose': 5000,
                       'saving_fitness': 100,
                       'is_restart': False}
            solver = Solver(problem, options)
            results = solver.optimize()
            pickle.dump(results, open(f.__name__ + '_' + str(d) + '.pickle', 'wb'))
        plot(f.__name__)
