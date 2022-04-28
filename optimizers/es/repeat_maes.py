"""Repeat Fig. 4 (Ellipsoid N=3 and N=30) from the following paper:
    Beyer, H.G. and Sendhoff, B., 2017.
    Simplify your covariance matrix adaptation evolution strategy.
    IEEE Transactions on Evolutionary Computation, 21(5), pp.746-759.
    https://ieeexplore.ieee.org/document/7875115
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from optimizers.es.es import ES
from optimizers.es.maes import MAES
from benchmarks.base_functions import ellipsoid

sns.set_theme(style='darkgrid')


class Fig4(MAES):
    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fit = ES.optimize(self, fitness_function)
        z, d, mean, s, tm, y = self.initialize()
        f, stepsize = [], []  # for plotting data
        while True:
            z, d, y = self.iterate(z, d, mean, tm, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fit.extend(y)
            f.append(np.min(y))
            stepsize.append(self.sigma)
            if self._check_terminations():
                break
            mean, s, tm = self._update_distribution(z, d, mean, s, tm, y)
            self._n_generations += 1
            self._print_verbose_info(y)
            z, d, mean, s, tm, y = self.restart_initialize(z, d, mean, s, tm, y)
        res = self._collect_results(fit, mean)
        res['s'] = s
        res['f'] = f
        res['stepsize'] = stepsize
        return res


if __name__ == '__main__':
    for ndim_problem in [3, 30]:
        problem = {'fitness_function': ellipsoid,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -10 * np.ones((ndim_problem,)),
                   'upper_boundary': 10 * np.ones((ndim_problem,))}
        options = {'fitness_threshold': 1e-15,
                   'seed_rng': 2022,  # not given in the original paper
                   'x': np.ones((ndim_problem,)),
                   'sigma': 1.0,
                   'stagnation': np.Inf}
        maes = Fig4(problem, options)
        results = maes.optimize()
        plt.figure()
        plt.yscale('log')
        plt.yticks(ticks=[1e-15, 1e-10, 1e-5, 1e0, 1e5])
        plt.plot(np.arange(len(results['f'])) + 1, results['f'], 'r-')
        plt.plot(np.arange(len(results['stepsize'])) + 1, results['stepsize'], 'm-')
        plt.xlabel('g')
        plt.ylabel('sigma and f(y)')
        plt.title('Ellipsoid N = ' + str(ndim_problem))
        plt.show()
