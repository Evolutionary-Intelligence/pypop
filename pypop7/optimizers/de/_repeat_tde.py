"""Repeat the following paper for `TDE`:
    Fan, H.Y. and Lampinen, J., 2003.
    A trigonometric mutation operation to differential evolution.
    Journal of Global Optimization, 27(1), pp.105-129.
    https://link.springer.com/article/10.1023/A:1024653025686

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/tde

    Luckily our code could repeat the data reported in the original paper *well*.
    Therefore, we argue that the repeatability of `TDE` could be **well-documented**.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import ackley, rastrigin
from pypop7.optimizers.de.tde import TDE as Solver


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    for f in [ackley, rastrigin]:
        plt.figure()
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        problem, options = {}, {}
        if f.__name__ == 'ackley':
            ndim_problem = 30
            problem = {'fitness_function': f,
                       'ndim_problem': ndim_problem,
                       'lower_boundary': -20 * np.ones((ndim_problem,)),
                       'upper_boundary': 30 * np.ones((ndim_problem,))}
            options = {'max_runtime': 45,
                       'seed_rng': 0,  # undefined in the original paper
                       'saving_fitness': 1,
                       'verbose': 2000}
        elif f.__name__ == 'rastrigin':
            ndim_problem = 20
            problem = {'fitness_function': f,
                       'ndim_problem': ndim_problem,
                       'lower_boundary': -5.12 * np.ones((ndim_problem,)),
                       'upper_boundary': 5.12 * np.ones((ndim_problem,))}
            options = {'max_runtime': 30,
                       'seed_rng': 0,  # undefined in the original paper
                       'saving_fitness': 1,
                       'verbose': 2000}
        solver = Solver(problem, options)
        results = solver.optimize()
        results['fitness'][:, 0] *= results['runtime']/results['n_function_evaluations']
        if f.__name__ == 'ackley':
            plt.ylim([0, 20])
            plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
            plt.xlim([0, 45])
            plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
            plt.ylabel("f1(X)")
        elif f.__name__ == 'rastrigin':
            plt.ylim([0, 160])
            plt.yticks([0, 20, 40, 60, 80, 100, 120, 140, 160])
            plt.xlim([0, 30])
            plt.xticks([0, 5, 10, 15, 20, 25, 30])
            plt.ylabel("f2(x)")
        plt.plot(results['fitness'][:, 0], results['fitness'][:, 1], color='black')
        plt.xlabel("CPU Time (Seconds)")
        plt.title(f.__name__.capitalize())
        plt.show()
