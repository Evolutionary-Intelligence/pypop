"""Repeat the following paper for `BES`:
    Gao, K. and Sener, O., 2022, June.
    Generalizing Gaussian Smoothing for Random Search.
    In International Conference on Machine Learning (pp. 7077-7101). PMLR.
    https://proceedings.mlr.press/v162/gao22f.html
    https://icml.cc/media/icml-2022/Slides/16434.pdf

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/bes

    Since the current implementation is only a *simplified* version of the original BES algorithm without noisy
    function (fitness) evaluations, its repeatability **cannot** be guaranteed.

    However, we found that the current implementation could show *very similar* performance on the same
    benchmark functions (after removing its noise part). The resulting much less (>100x) number of function
    evaluations needed may be well explained by the much easier fitness landscape (i.e., without noisiness).
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import sphere, rosenbrock
from pypop7.optimizers.rs.bes import BES


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')

    problem = {'fitness_function': sphere,
               'ndim_problem': 100}
    options = {'max_function_evaluations': 100*101,
               'seed_rng': 0,  # undefined in the original paper
               'x': np.random.default_rng(1).standard_normal(size=100),
               'n_individuals': 10,
               'saving_fitness': 101}
    bes = BES(problem, options)
    results = bes.optimize()
    print(results)
    plt.figure()
    plt.xlabel('Rounds')
    plt.xticks([0, 20, 40, 60, 80, 100])
    plt.ylabel('Oracle')
    plt.yticks([0, 20, 40, 60, 80, 100])
    plt.plot(results['fitness'][:, 0]/101, results['fitness'][:, 1], 'k')
    plt.show()

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': 100,
               'lower_boundary': -2*np.ones((100,)),
               'upper_boundary': 2*np.ones((100,))}
    options = {'max_function_evaluations': 100*101,
               'seed_rng': 2,  # undefined in the original paper
               'x': np.random.default_rng(3).standard_normal(size=100),
               'n_individuals': 10,
               'c': 0.1,
               'lr': 0.000001,
               'saving_fitness': 101}
    bes = BES(problem, options)
    results = bes.optimize()
    print(results)
    plt.figure()
    plt.xlabel('Rounds')
    plt.xticks([0, 20, 40, 60, 80, 100])
    plt.ylabel('Oracle')
    plt.yticks([10000, 20000, 30000, 40000])
    plt.plot(results['fitness'][:, 0]/101, results['fitness'][:, 1], 'k')
    plt.show()
