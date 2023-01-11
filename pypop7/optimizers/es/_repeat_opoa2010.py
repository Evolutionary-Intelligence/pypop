"""Repeat the following paper for `OPOA2010`:
    Arnold, D.V. and Hansen, N., 2010, July.
    Active covariance matrix adaptation for the (1+1)-CMA-ES.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 385-392). ACM.
    https://dl.acm.org/doi/abs/10.1145/1830483.1830556

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/opoa2010

    Luckily our Python code could repeat the data reported in the paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import sphere, ellipsoid, cigar, discus, cigar_discus
from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.opoa2010 import OPOA2010


class Fig1(OPOA2010):
    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        mean, y, a, a_i, best_so_far_y, p_s, p_c = self.initialize(args)
        sigmas = [self.sigma]  # for plotting
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            mean, y, a, a_i, best_so_far_y, p_s, p_c = self.iterate(
                mean, a, a_i, best_so_far_y, p_s, p_c, args)
            sigmas.append(self.sigma)  # for plotting
            self._n_generations += 1
            if self.is_restart:
                mean, y, a, a_i, best_so_far_y, p_s, p_c = self.restart_reinitialize(
                    mean, y, a, a_i, best_so_far_y, p_s, p_c, fitness, args)
        res = self._collect(fitness, y, mean)
        res['sigmas'] = sigmas  # for plotting
        return res


def plot(function, dim):
    res = pickle.load(open(function + '_' + str(dim) + '.pickle', 'rb'))
    plt.figure()
    plt.yscale('log')
    plt.ylim([1e-9, 1e6])
    plt.yticks([1e-9, 1e-6, 1e-3, 1e0, 1e3, 1e6])
    plt.xticks([0, 1e3, 2e3, 3e3, 4e3, 5e3, 6e3])
    plt.xlim([0, 6e3])
    plt.plot(res['fitness'][:, 0], res['fitness'][:, 1],
             color='r', label='function value f(x)')
    plt.plot(np.arange(len(res['sigmas'])), res['sigmas'],
             color='g', label=r'mutation strength $\sigma$')
    plt.xlabel("time t")
    plt.legend()
    plt.show()


def plot2(function, evaluation, dim):
    plt.xticks(rotation=30)
    plt.bar(function, evaluation, color='g', width=0.5)
    if dim == 2:
        plt.ylim([0, 6e2])
        plt.yticks([0, 2e2, 4e2, 6e2])
    elif dim == 4:
        plt.ylim([0, 1500])
        plt.yticks([0, 5e2, 1e3, 1.5e3])
    elif dim == 10:
        plt.ylim([0, 7.2e3])
        plt.yticks([0, 2.4e3, 4.8e3, 7.2e3])
    elif dim == 20:
        plt.ylim([0, 2.4e4])
        plt.yticks([0, 8e3, 1.6e4, 2.4e4])
    elif dim == 40:
        plt.ylim([0, 8.4e4])
        plt.yticks([0, 2.8e4, 5.6e4, 8.4e4])
    plt.title("n = " + str(dim))
    plt.show()


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    d = 10
    problem = {'fitness_function': discus,
               'ndim_problem': d}
    options = {'fitness_threshold': 1e-9,
               'seed_rng': 0,  # not given in the original paper
               'x': np.random.default_rng(1).standard_normal((d,)),  # mean
               'sigma': 0.1,
               'saving_fitness': 1}
    solver = Fig1(problem, options)
    results = solver.optimize()
    pickle.dump(results, open(discus.__name__ + '_' + str(d) + '.pickle', 'wb'))
    plot(discus.__name__, d)

    functions = [sphere, ellipsoid, cigar, discus, cigar_discus]
    for d in [10, 20, 40]:
        evaluations = []
        for f in functions:
            problem = {'fitness_function': f,
                       'ndim_problem': d}
            options = {'fitness_threshold': 1e-10,
                       'seed_rng': 0,
                       'x': np.random.default_rng(1).standard_normal((d,)),  # mean
                       'sigma': 0.1,
                       'is_restart': False,
                       'saving_fitness': 1}
            solver = OPOA2010(problem, options)
            results = solver.optimize()
            evaluations.append(results['n_function_evaluations'])
        plot2([f.__name__ for f in functions], evaluations, d)
