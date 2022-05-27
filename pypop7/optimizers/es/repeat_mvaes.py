import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pypop7.benchmarks.base_functions import sphere
from pypop7.optimizers.es.mvaes import MVAES as Solver

sns.set_theme(style='darkgrid')


if __name__ == '__main__':
    ndim_problem = 20
    for f in [sphere]:
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
        solver = Solver(problem, options)
        results = solver.optimize()
        print(results)
        plt.plot(results['fitness'][:, 0], results['fitness'][:, 1], 'k')
        options['n_individuals'] = 35
        options['n_parents'] = 5
        solver = Solver(problem, options)
        results = solver.optimize()
        print(results)
        plt.plot(results['fitness'][:, 0], results['fitness'][:, 1], 'k--')
        plt.yscale('log')
        plt.xticks(ticks=[0, 1000, 2000, 3000, 4000, 5000, 6000])
        plt.yticks(ticks=[1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4])
        plt.title('f_1 (Sphere)')
        plt.show()
