"""Repeat Fig. 4 (Sphere, Cigar, Tablet (aka Discus), Ellipsoid) from the following paper:
    Beyer, H.G. and Sendhoff, B., 2017.
    Simplify your covariance matrix adaptation evolution strategy.
    IEEE Transactions on Evolutionary Computation, 21(5), pp.746-759.
    https://ieeexplore.ieee.org/document/7875115
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from pypop7.benchmarks.base_functions import sphere, cigar, discus, ellipsoid
from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.maes import MAES

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


def plot(function, ndim):
    plt.figure()
    result = read_pickle(function, ndim)
    print(result)
    plt.yscale('log')
    plt.plot(np.arange(len(results['f'])), results['f'], 'r-')
    plt.plot(np.arange(len(results['stepsize'])), results['stepsize'], 'm-')
    if function == 'sphere':
        plt.ylim([1e-15, 1e2])
        plt.yticks([1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2])
        if ndim == 3:
            plt.xlim([0, 150])
            plt.xticks([0, 20, 40, 60, 80, 100, 120, 140])
        elif ndim == 30:
            plt.xlim([0, 600])
            plt.xticks([0, 100, 200, 300, 400, 500, 600])
    elif function == 'cigar':
        plt.ylim([1e-15, 1e8])
        plt.yticks([1e-15, 1e-10, 1e-5, 1e0, 1e5])
        if ndim == 3:
            plt.xlim([0, 250])
            plt.xticks([0, 50, 100, 150, 200, 250])
        elif ndim == 30:
            plt.xlim([0, 1400])
            plt.xticks([0, 200, 400, 600, 800, 1000, 1200, 1400])
    elif function == 'discus':
        plt.ylim([1e-15, 1e6])
        plt.yticks([1e-15, 1e-10, 1e-5, 1e0, 1e5])
        if ndim == 3:
            plt.xlim([0, 280])
            plt.xticks([0, 50, 100, 150, 200, 250])
        elif ndim == 30:
            plt.xlim([0, 3000])
            plt.xticks([0, 500, 1000, 1500, 2000, 2500])
    elif function == 'ellipsoid':
        plt.ylim([1e-15, 1e6])
        plt.yticks([1e-15, 1e-10, 1e-5, 1e0, 1e5])
        if ndim == 3:
            plt.xlim([0, 250])
            plt.xticks([0, 50, 100, 150, 200, 250])
        elif ndim == 30:
            plt.xlim([0, 3000])
            plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
    plt.xlabel("g")
    plt.ylabel(r"$\sigma^{(g)}$  and  $f(y^{(g)})$")
    plt.title(function.capitalize() + " N = " + str(ndim))
    plt.show()


class Fig4(MAES):
    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fit = ES.optimize(self, fitness_function)
        z, d, mean, s, tm, y = self.initialize()
        best_f, stepsize = [], []  # for plotting data
        while True:
            z, d, y = self.iterate(z, d, mean, tm, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fit.extend(y)
            best_f.append(np.min(y))
            stepsize.append(self.sigma)
            if self._check_terminations():
                break
            mean, s, tm = self._update_distribution(z, d, mean, s, tm, y)
            self._n_generations += 1
            self._print_verbose_info(y)
        res = self._collect_results(fit, mean)
        res['s'] = s
        res['f'] = best_f
        res['stepsize'] = stepsize
        return res


if __name__ == '__main__':
    for f in [sphere, cigar, discus, ellipsoid]:
        for dim in [3, 30]:
            problem = {'fitness_function': f,
                       'ndim_problem': dim}
            options = {'fitness_threshold': 1e-15,
                       'seed_rng': 2022,  # not given in the original paper
                       'x': np.ones((dim,)),
                       'sigma': 1.0,
                       'verbose_frequency': 10,
                       'record_fitness': True,
                       'record_fitness_frequency': 1}
            solver = Fig4(problem, options)
            results = solver.optimize()
            write_pickle(f.__name__, dim, results)
            plot(f.__name__, dim)
