"""Repeat Fig.1 in this paper
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
import time

from pypop7.optimizers.es.opoa import OPOA
from pypop7.optimizers.es.es import ES
from pypop7.benchmarks.base_functions import discus

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
    index = np.arange(1, len(result['sigmas']) + 1, 1)
    plt.figure()
    plt.yscale('log')
    plt.ylim([1e-9, 1e6])
    plt.yticks([1e-9, 1e-6, 1e-3, 1e0, 1e3, 1e6])
    plt.xticks([0, 1e3, 2e3, 3e3, 4e3, 5e3, 6e3])
    plt.xlim([0, 6e3])
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='r', label='function value f(x)')
    plt.plot(index, result['sigmas'], color='g', label=r'mutation strength $\sigma$')
    plt.xlabel("function evaluations")
    plt.ylabel("function value")
    plt.title(function.capitalize() + " Dimension: " + str(problem_dim))
    plt.legend()
    plt.show()


class Fig1(OPOA):
    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        sigmas = []
        start_run = time.time()
        mean, y, a, a_i, best_so_far_y, p_s, p_c = self.initialize(args)
        fitness.append(y)
        while True:
            mean, y, a, a_i, best_so_far_y, p_s, p_c = self.iterate(
                args, mean, a, a_i, best_so_far_y, p_s, p_c)
            sigmas.append(self.sigma)
            if self.record_fitness:
                fitness.append(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                mean, y, a, a_i, best_so_far_y, p_s, p_c = self.restart_initialize(
                    args, mean, y, a, a_i, best_so_far_y, p_s, p_c, fitness)
        results = self._collect_results(fitness, mean)
        results['runtime'] = time.time() - start_run
        results['sigmas'] = sigmas
        return results


if __name__ == '__main__':
    for f in [discus]:
        for d in [10]:
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
            solver = Fig1(problem, options)
            results = solver.optimize()
            write_pickle(f.__name__, d, results)
            plot(f.__name__, d)
