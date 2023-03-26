"""Repeat the following paper for `G3PCX`:
    Deb, K., Anand, A. and Joshi, D., 2002.
    A computationally efficient evolutionary algorithm for real-parameter optimization.
    Evolutionary Computation, 10(4), pp.371-395.
    https://direct.mit.edu/evco/article-abstract/10/4/371/1136/A-Computationally-Efficient-Evolutionary-Algorithm
    https://www.egr.msu.edu/~kdeb/codes/g3pcx/g3pcx.tar  (See the original C source code.)

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/repeatability/g3pcx/_repeat_g3pcx.png

    Luckily our Python code could repeat the data reported in the original paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import schwefel12
from pypop7.optimizers.ga.g3pcx import G3PCX as Solver


if __name__ == '__main__':
    ndim_problem = 20
    for f in [schwefel12]:
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem,
                   'initial_lower_boundary': -10*np.ones((ndim_problem,)),
                   'initial_upper_boundary': -5*np.ones((ndim_problem,))}
        options = {'max_function_evaluations': 50000,
                   'fitness_threshold': 1e-20,
                   'seed_rng': 0,
                   'saving_fitness': 1}
        solver = Solver(problem, options)
        results = solver.optimize()

        sns.set_theme(style='darkgrid')
        plt.figure()
        fitness = results['fitness']
        plt.plot(fitness[:, 0], fitness[:, 1], 'k')
        plt.xticks([0, 50000, 100000, 150000, 200000])
        plt.xlabel('Function Evaluations')
        plt.yscale('log')
        plt.yticks([1e-20, 1e-15, 1e-10, 1e-5, 1e0, 1e5])
        plt.ylabel('Best Fitness')
        plt.show()
