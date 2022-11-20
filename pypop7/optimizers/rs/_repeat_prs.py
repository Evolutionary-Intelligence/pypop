"""Repeat the following paper for `PRS`:
    Hansen, N., Arnold, D.V. and Auger, A., 2015.
    Evolution strategies.
    In Springer Handbook of Computational Intelligence (pp. 871-898).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007%2F978-3-662-43505-2_44

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/repeatability/prs/_repeat_prs.png

    Luckily our code could repeat the data reported in the original paper *well*.
    Therefore, we argue that the repeatability of `PRS` could be **well-documented**.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import sphere
from pypop7.optimizers.rs.prs import PRS


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    plt.figure()
    ndim = 10
    for i in range(3):
        problem = {'fitness_function': sphere,
                   'ndim_problem': ndim,
                   'upper_boundary': 1 * np.ones((ndim,)),
                   'lower_boundary': -0.2 * np.ones((ndim,))}
        options = {'max_function_evaluations': 1500,
                   'seed_rng': i,  # undefined in the original paper
                   'saving_fitness': 1}
        prs = PRS(problem, options)
        results = prs.optimize()
        fitness = results['fitness']
        plt.plot(fitness[:, 0], fitness[:, 1], 'r')
    plt.xticks([0, 500, 1000, 1500])
    plt.xlim([0, 1500])
    plt.yscale('log')
    plt.yticks([1e-9, 1e-6, 1e-3, 1e0])
    plt.show()
