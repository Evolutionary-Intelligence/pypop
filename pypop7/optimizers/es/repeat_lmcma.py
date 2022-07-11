"""Repeat Fig.3, Fig.4 from the following paper:
    I. Loshchilov
    LM-CMA: an Alternative to L-BFGS for Large Scale Black-box Optimization
    Evolutionary Computation 2017; 25(1): 143-171
    https://direct.mit.edu/evco/article-abstract/25/1/143/1041/LM-CMA-An-Alternative-to-L-BFGS-for-Large-Scale?redirectedFrom=fulltext
"""
import sys
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pypop7.benchmarks.base_functions import ellipsoid, rosenbrock, discus, cigar, different_powers
from pypop7.benchmarks.rotated_functions import generate_rotation_matrix
from pypop7.benchmarks.rotated_functions import ellipsoid as rotated_ellipsoid
from pypop7.benchmarks.rotated_functions import rosenbrock as rotated_rosenbrock
from pypop7.benchmarks.rotated_functions import discus as rotated_discus
from pypop7.benchmarks.rotated_functions import cigar as rotated_cigar
from pypop7.benchmarks.rotated_functions import different_powers as rotated_different_powers
from pypop7.optimizers.core import optimizer
from pypop7.optimizers.es.lmcma import LMCMA as Solver

sys.modules['optimizer'] = optimizer
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
    plt.figure()
    result = read_pickle(function, problem_dim)
    print(result)
    plt.yscale('log')
    plt.xscale('log')
    if function in ['ellipsoid', 'rotated_ellipsoid']:
        plt.ylim([1e-10, 1e10])
        plt.yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8, 1e10])
        plt.xlim([1e2, 2e8])
        plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    elif function in ['rosenbrock', 'rotated_rosenbrock']:
        plt.ylim([1e-10, 1e9])
        plt.yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
        plt.xlim([1e2, 1e9])
        plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    elif function in ['discus', 'rotated_discus']:
        plt.ylim([1e-10, 1e8])
        plt.yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
        plt.xlim([1e0, 1e8])
        plt.xticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    elif function in ['cigar', 'rotated_cigar']:
        plt.ylim([1e-10, 1e12])
        plt.yticks([1e-10, 1e-5, 1e0, 1e5, 1e10])
        plt.xlim([1e2, 1e7])
        plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6])
    elif function in ['different_powers', 'rotated_different_powers']:
        plt.ylim([1e-10, 1e7])
        plt.yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6])
        plt.xlim([1e0, 1e8])
        plt.xticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='blue')
    plt.xlabel("Function Evaluations")
    plt.ylabel("Objective Function")
    plt.title(function.capitalize())
    plt.savefig('lmcma/result_' + function + '.png')
    plt.show()


if __name__ == '__main__':
    for f in [ellipsoid, rosenbrock, discus, cigar, different_powers, rotated_ellipsoid,
              rotated_rosenbrock, rotated_discus, rotated_cigar, rotated_different_powers]:
        for d in [512]:
            problem = {'fitness_function': f,
                       'ndim_problem': d,
                       'lower_boundary': -5 * np.ones((d,)),
                       'upper_boundary': 5 * np.ones((d,))}
            options = {'max_function_evaluations': 2e7,
                       'max_runtime': 3600,
                       'fitness_threshold': 1e-10,
                       'seed_rng': 0,  # undefined in the original paper
                       'x': 4 * np.ones((d,)),  # mean
                       'sigma': 3.0,
                       'verbose_frequency': 2e4,
                       'record_fitness': True,
                       'record_fitness_frequency': 1,
                       'is_restart': False}
            if f in [rotated_ellipsoid, rotated_rosenbrock, rotated_discus,
                              rotated_cigar, rotated_different_powers]:
                generate_rotation_matrix(f, d, 2022)
                function_name = 'rotated_' + f.__name__
            else:
                function_name = f.__name__
            solver = Solver(problem, options)
            results = solver.optimize()
            write_pickle(function_name, d, results)
            plot(function_name, d)
