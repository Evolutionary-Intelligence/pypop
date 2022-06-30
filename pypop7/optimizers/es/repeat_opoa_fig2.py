"""Repeat Fig.2 in this paper
    (1+1)-Active-CMA-ES (OPOA).
    Reference
    ---------
    Arnold, D.V. and Hansen, N., 2010, July.
    Active covariance matrix adaptation for the (1+1)-CMA-ES.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 385-392). ACM.
    https://dl.acm.org/doi/abs/10.1145/1830483.1830556
    """
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from pypop7.optimizers.es.opoa import OPOA as Solver
from pypop7.benchmarks.base_functions import sphere, ellipsoid, cigar, discus, cigar_discus

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


def plot(functions, evaluations, problem_dim):
    plt.xticks(rotation=30)
    plt.bar(functions, evaluations, color='g', width=0.5)
    if problem_dim == 2:
        plt.ylim([0, 6e2])
        plt.yticks([0, 2e2, 4e2, 6e2])
    elif problem_dim == 4:
        plt.ylim([0, 1500])
        plt.yticks([0, 5e2, 1e3, 1.5e3])
    elif problem_dim == 10:
        plt.ylim([0, 7.2e3])
        plt.yticks([0, 2.4e3, 4.8e3, 7.2e3])
    elif problem_dim == 20:
        plt.ylim([0, 2.4e4])
        plt.yticks([0, 8e3, 1.6e4, 2.4e4])
    elif problem_dim == 40:
        plt.ylim([0, 8.4e4])
        plt.yticks([0, 2.8e4, 5.6e4, 8.4e4])
    plt.title(" Dimension: " + str(problem_dim))
    plt.show()


if __name__ == '__main__':
    functions = [sphere, ellipsoid, cigar, discus, cigar_discus]
    function_name = ['sphere', 'ellipsoid', 'cigar', 'discus', 'cigdis']
    for d in [10, 20, 40]:
        evaluations = []
        for f in functions:
            print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
            problem = {'fitness_function': f,
                       'ndim_problem': d,
                       'lower_boundary': -5 * np.ones((d,)),
                       'upper_boundary': 5 * np.ones((d,))}
            options = {'max_function_evaluations': 2e6,
                       'fitness_threshold': 1e-9,
                       'max_runtime': 3600,  # 1 hours
                       'seed_rng': 0,
                       'x': 4 * np.ones((d,)),  # mean
                       'sigma': 0.1,
                       'verbose_frequency': 2000,
                       'record_fitness': True,
                       'is_restart': False,
                       'record_fitness_frequency': 1}
            solver = Solver(problem, options)
            results = solver.optimize()
            evaluations.append(results['n_function_evaluations'])
        plot(function_name, evaluations, d)
