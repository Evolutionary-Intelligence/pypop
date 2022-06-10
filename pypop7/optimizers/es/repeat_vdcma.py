"""Repeat Figure 1 (left sub-figure) from the following paper:
    Akimoto, Y., Auger, A. and Hansen, N., 2014, July.
    Comparison-based natural gradient optimization in high dimension.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 373-380). ACM.
    https://dl.acm.org/doi/abs/10.1145/2576768.2598258

    See the official Python version from Akimoto:
    https://gist.github.com/youheiakimoto/08b95b52dfbf8832afc71dfff3aed6c8
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.vdcma import VDCMA
from pypop7.benchmarks.base_functions import sphere

sns.set_theme(style='darkgrid')


def _rot_cigar(x):
    u = np.ones((len(x),)) / np.sqrt(len(x))  # unit vector
    y = np.power(10, 6) * sphere(x) + (1 - np.power(10, 6)) * np.power(np.dot(x, u), 2)
    return y


def _ell_cig(x):
    dim = len(x)
    d_ell = np.zeros((dim, dim))
    for i in range(dim):
        d_ell[i][i] = np.power(10, 3 * i / dim)  # diagonal matrix
    y = _rot_cigar(np.dot(d_ell, x))
    return y


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
    plt.yscale('log')
    plt.xlim([0, 3.5e4])
    plt.xticks([1e4, 2e4, 3e4])
    plt.ylim([1e-10, 1e13])
    plt.yticks([1e-10, 1e-5, 1e0, 1e5, 1e10])
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='b', label=r'$min_i$f($x_i$)')
    plt.plot(result['fitness'][:, 0], result['sigma'], color='g', label=r'$\sigma$')
    plt.xlabel("No. Func. Evals")
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1))
    plt.show()


class Fig1(VDCMA):
    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness, sigma = ES.optimize(self, fitness_function), []
        d, v, p_s, p_c, z, zz, x, mean, y = self.initialize()
        while True:
            z, zz, x, y = self.iterate(d, z, zz, x, mean, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y)
                sigma.extend(np.ones((len(y),)) * self.sigma)
            if self._check_terminations():
                break
            mean, p_s, p_c, v, d = self._update_distribution(d, v, p_s, p_c, zz, x, y)
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                d, v, p_s, p_c, z, zz, x, mean, y = self.restart_initialize(d, v, p_s, p_c, z, zz, x, mean, y)
        res = self._collect_results(fitness, mean)
        res['d'] = d
        res['v'] = v
        res['sigma'] = sigma
        return res


if __name__ == '__main__':
    # Fig. 1 (left sub-figure)
    ndim_problem = 50
    print('*' * 7 + ' ' + _ell_cig.__name__ + ' ' + '*' * 7)
    problem = {'fitness_function': _ell_cig,
               'ndim_problem': ndim_problem,
               'lower_boundary': -5 * np.ones((ndim_problem,)),
               'upper_boundary': 5 * np.ones((ndim_problem,))}
    options = {'fitness_threshold': 1e-10,
               'seed_rng': 0,  # not given in the original paper
               'mean': 3 + 2 * np.random.default_rng(0).standard_normal((ndim_problem,)),
               'sigma': 2,
               'verbose_frequency': 200,
               'record_fitness': True,
               'record_fitness_frequency': 1}
    solver = Fig1(problem, options)
    results = solver.optimize()
    write_pickle(_ell_cig.__name__, ndim_problem, results)
    plot(_ell_cig.__name__, ndim_problem)
