"""Repeat Figure 6 in paper
    Latent Action Monto Carlo Tree Search(LA-MCTS)
    Reference
    --------------
    L. Wang, R. Fonseca, Y. Tian
    Learning Search Space Partition for Black-box Optimization using Monte Carlo Tree Search
    NeurIPS 2020
    https://proceedings.neurips.cc/paper/2020/hash/e2ce14e81dba66dbff9cbc35ecfdb704-Abstract.html
    Code of paper:
    https://github.com/facebookresearch/LaMCTS/tree/main/LA-MCTS
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from pypop7.benchmarks.base_functions import ackley, rosenbrock
from pypop7.optimizers.bo.lamcts import LAMCTS as Solver

sns.set_theme(style='darkgrid')


def read_pickle(function, ndim):
    file = function + '_' + str(ndim) + '.pickle'
    with open(file, 'rb') as handle:
        result = pickle.load(handle)
        return result


def write_pickle(function, ndim, result):
    file = open(function + '_' + str(ndim) + '.pickle', 'wb')
    pickle.dump(result, file)
    file.close()


def plot(function, problem_dim):
    result = read_pickle(function, problem_dim)
    print("1")
    plt.figure()
    if function == "ackley":
        plt.ylim([0, 20])
        plt.yticks([0, 5, 10, 15])
    elif function == "rosenbrock":
        plt.yscale('log')
        plt.ylim([1e1, 1e8])
        plt.yticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    if problem_dim == 20:
        plt.xlim([0, 1000])
        plt.xticks([0, 200, 400, 600, 800, 1000])
    elif problem_dim == 100:
        plt.xlim([0, 10000])
        plt.yticks([0, 2e3, 4e3, 6e3, 8e3, 1e4])
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='orange')
    plt.xlabel("# samples")
    plt.ylabel("f(x)")
    plt.title(function.capitalize() + "-" + str(problem_dim) + "d")
    plt.show()


if __name__ == "__main__":
    for f in [ackley, rosenbrock]:
        for d in [20, 100]:
            print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
            problem = {'fitness_function': f,
                       'ndim_problem': d,
                       'lower_boundary': -5 * np.ones((d,)),
                       'upper_boundary': 10 * np.ones((d,))}
            options = {'max_function_evaluations': 1000,
                       'fitness_threshold': 1e-5,
                       'seed_rng': 0,
                       'n_individuals': 40,  # the number of random samples used in initialization
                       'Cp': 1,  # Cp for MCTS
                       'leaf_size': 10,  # tree leaf size
                       'kernel_type': "rbf",  # used in SVM
                       'gamma_type': "auto",  # used in SVM
                       'solver_type': 'bo',  # solver can be bo or turbo
                       'verbose_frequency': 1,
                       'record_fitness': True,
                       'record_fitness_frequency': 1}
            if f.__name__ == "rosenbrock":
                problem['lower_boundary'] = -10 * np.ones((d,))
            if d == 100:
                options['max_function_evaluations'] = 1e4
            solver = Solver(problem, options)
            results = solver.optimize()
            print(results)
            write_pickle(f.__name__, d, results)
            plot(f.__name__, d)
