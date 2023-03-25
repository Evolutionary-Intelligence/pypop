"""This is a simple demo for the popular Lennard-Jones clustering optimization problem
  from the chemistry field:

  https://esa.github.io/pagmo2/docs/cpp/problems/lennard_jones.html
  https://esa.github.io/pygmo2/install.html
"""
import pygmo as pg  # need to be installed: https://esa.github.io/pygmo2/install.html
from pypop7.optimizers.de.cde import CDE  # https://pypop.readthedocs.io/en/latest/de/cde.html
from pypop7.optimizers.de.jade import JADE  # https://pypop.readthedocs.io/en/latest/de/jade.html
import seaborn as sns
import matplotlib.pyplot as plt


# see https://esa.github.io/pagmo2/docs/cpp/problems/lennard_jones.html for the fitness function
prob = pg.problem(pg.lennard_jones(150))
print(prob)  # 444-dimensional


def energy_func(x):  # wrapper to obtain fitness of type `float`
    return float(prob.fitness(x))


if __name__ == '__main__':
    results = []  # to save all optimization results from different optimizers
    for DE in [CDE, JADE]:
        problem = {'fitness_function': energy_func,
                   'ndim_problem': 444,
                   'upper_boundary': prob.get_bounds()[1],
                   'lower_boundary': prob.get_bounds()[0]}
        options = {'max_function_evaluations': 400000,
                   'seed_rng': 2022,  # for repeatability
                   'saving_fitness': 1,  # to save all fitness generated during optimization
                   'boundary': True}  # for JADE (but not for CDE)
        solver = DE(problem, options)
        results.append(solver.optimize())
        print(results[-1])

    sns.set_theme(style='darkgrid')
    plt.figure()
    for label, res in zip(['CDE', 'JADE'], results):
        # starting 250000 can avoid excessively high values generated during the early stage
        #   to disrupt convergence curves
        plt.plot(res['fitness'][250000:, 0], res['fitness'][250000:, 1], label=label)

    plt.legend()
    plt.show()
