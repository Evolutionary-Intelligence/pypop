"""This is a simple demo that optimizes the Bragg mirrors structure, modeled in the following paper:
    Bennet, P., Centeno, E., Rapin, J., Teytaud, O. and Moreau, A., 2020.
    The photonics and ARCoating testbeds in NeverGrad.
    https://hal.uca.fr/hal-02613161v1
"""
import numpy as np
import matplotlib.pyplot as plt
from nevergrad.functions.photonics.core import Photonics

from pypop7.optimizers.pso.clpso import CLPSO  # https://pypop.readthedocs.io/en/latest/pso/clpso.html
from pypop7.optimizers.de.jade import JADE  # https://pypop.readthedocs.io/en/latest/de/jade.html


if __name__ == '__main__':
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = '12'

    labels = ['CLPSO', 'JADE']
    for i, Opt in enumerate([CLPSO, JADE]):
        ndim_problem = 10  # dimension of objective function
        half = int(ndim_problem/2)
        func = Photonics("bragg", ndim_problem)
        problem = {'fitness_function': func,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': np.hstack((2*np.ones(half), 30*np.ones(half))),
                   'upper_boundary': np.hstack((3*np.ones(half), 180*np.ones(half)))}
        options = {'max_function_evaluations': 50000,
                   'n_individuals': 200,
                   'is_bound': True,
                   'seed_rng': 0,
                   'saving_fitness': 1,
                   'verbose': 200}
        solver = Opt(problem, options)
        results = solver.optimize()
        res = results['fitness']
        plt.plot(res[:, 0], res[:, 1], linewidth=2.0, linestyle='-', label=labels[i])
    plt.legend()
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Fitness (to be Minimized)')
    plt.title('Bragg Mirrors Structure')
    plt.savefig('photonics_optimization.png')
