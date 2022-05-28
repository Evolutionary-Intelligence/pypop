import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pypop7.benchmarks.base_functions import cigar, ellipsoid, rosenbrock, sphere
from pypop7.optimizers.es.mvaes import MVAES as Solver

sns.set_theme(style='darkgrid')


if __name__ == '__main__':
    ndim_problem = 20
    for f in [cigar, ellipsoid, rosenbrock, sphere]:
        plt.figure()
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem}
        options = {'max_function_evaluations': 1e5,
                   'fitness_threshold': 1e-10,
                   'seed_rng': 0,
                   'x': np.ones((ndim_problem,)),
                   'n_individuals': 10,
                   'n_parents': 1,
                   'sigma': 1,
                   'verbose_frequency': 200,
                   'record_fitness': True,
                   'record_fitness_frequency': 1,
                   'is_restart': False}
        if f == ellipsoid:
            options['max_function_evaluations'] = 5e5
        solver = Solver(problem, options)
        results = solver.optimize()
        print(results)
        plt.plot(results['fitness'][:, 0], results['fitness'][:, 1], 'k', label=r'$\mu$=1, $\lambda$=10, no recombination')
        options['n_individuals'] = 35
        options['n_parents'] = 5
        solver = Solver(problem, options)
        results = solver.optimize()
        print(results)
        plt.plot(results['fitness'][:, 0], results['fitness'][:, 1], 'k--', label=r'$\mu$=5, $\lambda$=35, recombination')
        plt.yscale('log')
        if f == cigar:
            plt.xticks(ticks=[0, 1e4, 2e4, 3e4, 4e4, 5e4])
            plt.xlim([0, 5e4])
        elif f == ellipsoid:
            plt.xticks(ticks=[0, 5e4, 1e5, 1.5e5, 2e5, 2.5e5, 3e5, 3.5e5])
            plt.xlim([0, 3.5e5])
        elif f == sphere:
            plt.xticks(ticks=[0, 1000, 2000, 3000, 4000, 5000, 6000])
            plt.xlim(([0, 6000]))
        elif f == rosenbrock:
            plt.xticks(ticks=[0, 2e4, 4e4, 6e4, 8e4, 1e5])
            plt.xlim([0, 1e5])
        plt.ylim([1e-10, 1e4])
        plt.yticks(ticks=[1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4])
        plt.xlabel('function evaluations')
        plt.ylabel('fitness')
        plt.legend(loc="upper right", bbox_to_anchor=(1, 1))
        plt.title(f.__name__)
        plt.show()
