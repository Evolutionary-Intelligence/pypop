"""Repeat the following paper for `SAMAES`:
    Beyer, H.G., 2020, July.
    Design principles for matrix adaptation evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation Companion (pp. 682-700). ACM.
    https://dl.acm.org/doi/abs/10.1145/3377929.3389870

    Luckily our Python code could repeat the data reported in the paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.optimizers.es.samaes import SAMAES


# set parameters for lens shape optimization
weight = 0.9  # weight of focus function
r = 7  # radius of lens
h = 1  # trapezoidal slices of height
b = 20  # distance between lens and object
eps = 1.5  # refraction index
d_init, sigma = 3.0, 1.0  # initialization


# define objective function to be minimized
def f_lens(x):  # refer to [Beyer, 2020, ACM-GECCO] for the mathematical details
    n = len(x)
    focus = r - ((h*np.arange(1, n) - 0.5) + b/h*(eps - 1)*np.transpose(np.abs(x[1:]) - np.abs(x[:(n-1)])))
    mass = h*(np.sum(np.abs(x[1:(n-1)])) + 0.5*(np.abs(x[0]) + np.abs(x[n-1])))
    return weight*np.sum(focus**2) + (1.0 - weight)*mass


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    ndim_problem = 15  # dimension of objective function
    problem = {'fitness_function': f_lens,  # objective function
               'ndim_problem': ndim_problem,  # number of dimensionality of objective function
               'lower_boundary': -5*np.ones((ndim_problem,)),  # lower boundary of search range
               'upper_boundary': 5*np.ones((ndim_problem,))}  # upper boundary of search range
    options = {'max_function_evaluations': 1000*20,  # maximum of function evaluations
               'seed_rng': 1,  # seed of random number generation (for repeatability)
               'n_individuals': 20,
               'n_parents': 5,
               'x': d_init*np.ones((ndim_problem,)),  # initial mean of Gaussian search distribution
               'sigma': sigma,  # global step-size of Gaussian search distribution (not necessarily an optimal value)
               'saving_fitness': 1,  # to record best-so-far fitness every 50 function evaluations
               'is_restart': False}  # whether or not to run the (default) restart process
    results = SAMAES(problem, options).optimize()
    print(results)
    plt.plot(results['fitness'][:, 0]/20, results['fitness'][:, 1], 'b')
    plt.xlabel('g')
    plt.xticks([0, 200, 400, 600, 800, 1000])
    plt.xlim([0, 1000])
    plt.ylabel('f_lens')
    plt.yscale('log')
    plt.yticks([1e0, 1e1, 1e2, 1e3])
    plt.show()
