"""This is a simple demo that optimizes the Bragg mirrors structures, modeled in the following paper:
    Bennet, P., Centeno, E., Rapin, J., Teytaud, O. and Moreau, A., 2020.
    The photonics and ARCoating testbeds in Nevergrad.
    https://hal.uca.fr/hal-02613161v1
"""
import numpy as np
import matplotlib.pyplot as plt
from nevergrad.functions.photonics.core import Photonics

from pypop7.optimizers.pso.clpso import CLPSO

def plot(x, y, name):
    sub_figure = name  + '.png'
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = '12'

    plt.plot(x, y, color='black', linewidth=1.0, linestyle='-', label='CLPSO')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = '12'
    plt.legend()
    plt.xlabel('Fevals')
    plt.ylabel('Fitness')
    plt.title('Bragg mirrors structures')
    plt.savefig(sub_figure)


if __name__ == '__main__':
    ndim_problem = 10 # dimension of objective function
    half = int(ndim_problem/2)
    func = Photonics("bragg", ndim_problem)
    problem = {'fitness_function': func,
               'ndim_problem': ndim_problem,
               'lower_boundary': np.hstack((2*np.ones(half), 30*np.ones(half))),
               'upper_boundary': np.hstack((3*np.ones(half), 180*np.ones(half)))}
    options = {'max_function_evaluations': 50000,  # 100000
               'fitness_threshold': 1e-10,
               'n_individuals': 200,
               'is_bound': True,
               'seed_rng': 0,
               'saving_fitness': 200,
               'verbose': 200}
    solver = CLPSO(problem, options)
    results = solver.optimize()
    res = results['fitness']
    plot(res[:, 0], res[:, 1], 'photonics_optimization')
