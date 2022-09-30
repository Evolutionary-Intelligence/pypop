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
               'x': np.ones((100,)),
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
               'lower_boundary': -2 * np.ones((100,)),
               'upper_boundary': 2 * np.ones((100,))}
    options = {'max_function_evaluations': 100*101,
               'seed_rng': 0,  # undefined in the original paper
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
