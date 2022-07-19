"""Repeat Fig.5 Function T12 k=3, M=1000 in paper:
    A. Kaban, J. Bootkrajang, R. J. Durrant
    Towards Large Scale Continuous EDA: A Random Matrix Theory Perspective
    GECCO 2013: 383-390
"""
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import rosenbrock
from pypop7.optimizers.core.optimizer import Optimizer
from pypop7.optimizers.eda.rpeda import RPEDA

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


def plot(function):
    plt.figure()
    result = read_pickle(function, 1000)
    xs = np.arange(len(result))
    plt.yscale('log')
    plt.plot(xs, result, color='r')
    plt.xlim([0, 2e3])
    plt.ylim([1e3, 1e13])
    plt.xticks([200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    plt.yticks(ticks=[1e4, 1e6, 1e8, 1e10, 1e12])
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title(function.capitalize())
    plt.show()


class Fig5(RPEDA):
    def optimize(self, fitness_function=None):
        fitness = Optimizer.optimize(self, fitness_function)
        x, x_fit, y = self.initialize()
        best_fitness = []
        while True:
            order = np.argsort(y)
            for i in range(self.n_parents):
                x_fit[i] = x[order[i]]
            if self.record_fitness:
                fitness.extend(y)
            best_fitness.append(y[order[0]])
            if self._check_terminations():
                break
            x, y = self.iterate(x_fit, y)
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results, best_fitness


if __name__ == '__main__':
    ndim_problem = 1000
    for f in [rosenbrock]:
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -100 * np.ones((ndim_problem,)),
                   'upper_boundary': 100 * np.ones((ndim_problem,))}
        options = {'max_function_evaluations': 6e5,
                   'fitness_threshold': 1e-10,
                   'seed_rng': 0,
                   'k': 3,
                   'rpmSize': 1000,
                   'typeR': 'G',
                   'n_parents': 75,
                   'n_individuals': 300,
                   'verbose_frequency': 200,
                   'record_fitness': False,
                   'record_fitness_frequency': 1}
        solver = Fig5(problem, options)
        results, best_fitness = solver.optimize()
        write_pickle(f.__name__, ndim_problem, best_fitness)
        plot(f.__name__)
