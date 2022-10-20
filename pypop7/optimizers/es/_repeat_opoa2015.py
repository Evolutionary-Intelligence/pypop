"""Repeat the following paper for `OPOA2015`:
    Krause, O. and Igel, C., 2015, January.
    A more efficient rank-one covariance matrix update for evolution strategies.
    In Proceedings of ACM Conference on Foundations of Genetic Algorithms (pp. 129-136).
    https://dl.acm.org/doi/abs/10.1145/2725494.2725496

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/opoa2015

    Luckily our code could repeat the data reported in the original paper *well*.
    Therefore, we argue that the repeatability of `OPOA2015` could be **well-documented**.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.optimizers.es.opoa2015 import OPOA2015


def cigar(x):
    x = np.power(x, 2)
    y = 1e-3*x[0] + np.sum(x[1:])
    return y


def discus(x):  # also called tablet
    x = np.power(x, 2)
    y = x[0] + 1e-3*np.sum(x[1:])
    return y


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    colors = ['r', 'k', 'b', 'orange']
    # discus
    plt.figure()
    plt.xlim([0, 10000])
    plt.xticks([0, 2000, 4000, 6000, 8000, 10000])
    plt.xlabel('iterations')
    plt.ylim([1e-8, 1])
    plt.yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    plt.ylabel('function value')
    plt.yscale('log')
    for c, d in zip(colors, np.array([1, 2, 4, 8]) * 100):
        problem = {'fitness_function': discus,
                   'ndim_problem': d}
        options = {'seed_rng': 0,  # not given in the original paper
                   'x': np.random.default_rng(1).standard_normal((d,)),  # mean
                   'max_function_evaluations': 10000,
                   'sigma': 0.1,
                   'saving_fitness': 1,
                   'is_restart': False}
        solver = OPOA2015(problem, options)
        results = solver.optimize()
        print(results)
        plt.plot(results['fitness'][:, 0], results['fitness'][:, 1], color=c, label=f'n={d}')
    plt.legend()
    plt.show()
    # cigar
    plt.figure()
    plt.xlim([0, 10000])
    plt.xticks([0, 2000, 4000, 6000, 8000, 10000])
    plt.xlabel('iterations')
    plt.ylim([1e-5, 1000])
    plt.yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e2, 1e3])
    plt.ylabel('function value')
    plt.yscale('log')
    for c, d in zip(colors, np.array([1, 2, 4, 8]) * 100):
        problem = {'fitness_function': cigar,
                   'ndim_problem': d}
        options = {'seed_rng': 0,  # not given in the original paper
                   'x': np.random.default_rng(1).standard_normal((d,)),  # mean
                   'max_function_evaluations': 10000,
                   'sigma': 0.1,
                   'saving_fitness': 1,
                   'is_restart': False}
        solver = OPOA2015(problem, options)
        results = solver.optimize()
        print(results)
        plt.plot(results['fitness'][:, 0], results['fitness'][:, 1], color=c, label=f'n={d}')
    plt.legend()
    plt.show()
