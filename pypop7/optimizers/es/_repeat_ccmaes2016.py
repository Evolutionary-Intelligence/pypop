"""Repeat the following paper for `CCMAES2016`:
    Krause, O., Arbonès, D.R. and Igel, C., 2016.
    CMA-ES with optimal covariance update and storage complexity.
    Advances in Neural Information Processing Systems, 29, pp.370-378.
    https://proceedings.neurips.cc/paper/2016/hash/289dff07669d7a23de0ef88d2f7129e7-Abstract.html

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/ccmaes2016
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import sphere
from pypop7.optimizers.es.ccmaes2016 import CCMAES2016


def cigar(x):
    x = np.power(x, 2)
    y = 1e-6*x[0] + np.sum(x[1:])
    return y


def discus(x):  # also called tablet
    x = np.power(x, 2)
    y = x[0] + 1e-6*np.sum(x[1:])
    return y


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    dims = [4, 8, 16, 32, 64, 128, 256]
    for f in [sphere, cigar, discus]:
        plt.figure()
        plt.ylim([1e2, 1e4])
        plt.xscale('log')
        plt.yticks([1e2, 1e3, 1e4])
        plt.ylabel('iterations')
        plt.yscale('log')
        n_iterations = []
        for d in [4, 8, 16, 32, 64, 128, 256]:
            problem = {'fitness_function': f,
                       'ndim_problem': d,
                       'lower_boundary': np.zeros((d,)),
                       'upper_boundary': np.ones((d,))}
            options = {'seed_rng': 0,  # not given in the original paper
                       'fitness_threshold': 1e-14,
                       'sigma': 1 / 3,  # not given in the original paper
                       'saving_fitness': 1,
                       'is_restart': False}
            if f == sphere:
                options['x'] = np.random.default_rng(1).standard_normal((d,))  # mean
            solver = CCMAES2016(problem, options)
            results = solver.optimize()
            n_iterations.append(results['_n_generations'])
        print(n_iterations)
        plt.plot(dims, n_iterations, color='r')
        plt.semilogx(base=2)
        plt.xlim([4, 256])
        plt.xticks(dims)
        plt.show()
