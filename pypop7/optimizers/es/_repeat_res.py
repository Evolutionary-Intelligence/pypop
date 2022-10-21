"""Repeat the following paper for `RES`:
    Hansen, N., Arnold, D.V. and Auger, A., 2015.
    Evolution strategies.
    In Springer Handbook of Computational Intelligence (pp. 871-898).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007%2F978-3-662-43505-2_44

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/res

    Luckily our code could repeat the data reported in the original paper *well*.
    Therefore, we argue that the repeatability of `RES` could be **well-documented**.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import sphere
from pypop7.optimizers.es.res import RES


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    plt.figure()
    for i in range(3):
        problem = {'fitness_function': sphere,
                   'ndim_problem': 10}
        options = {'max_function_evaluations': 1500,
                   'seed_rng': i,  # undefined in the original paper
                   'saving_fitness': 1,
                   'x': np.ones((10,)),
                   'sigma': 1e-9,
                   'lr_sigma': 1.0/(1.0 + 10.0/3.0),
                   'is_restart': False}
        res = RES(problem, options)
        fitness = res.optimize()['fitness']
        plt.plot(fitness[:, 0], np.sqrt(fitness[:, 1]), 'b')  # sqrt for distance
    plt.xticks([0, 500, 1000, 1500])
    plt.xlim([0, 1500])
    plt.yticks([1e-9, 1e-6, 1e-3, 1e0])
    plt.yscale('log')
    plt.show()
