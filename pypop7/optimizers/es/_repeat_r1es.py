"""Repeat the following paper for `R1ES`:
    Li, Z. and Zhang, Q., 2018.
    A simple yet efficient evolution strategy for large-scale black-box optimization.
    IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
    https://ieeexplore.ieee.org/abstract/document/8080257

    All generated figures can be accessed via the following link:
    https://github.com/Evolutionary-Intelligence/pypop/tree/main/docs/repeatability/r1es

    Luckily our Python code could repeat the data reported in the paper *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import cigar, ellipsoid, discus, rosenbrock
from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.r1es import R1ES


def plot(function, ndim):
    plt.figure()
    result = pickle.load(open(function + '_' + str(ndim) + '.pickle', 'rb'))
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color='purple')
    plt.xlim([1e2, 1e8])
    plt.xticks([1e2, 1e4, 1e6, 1e8])
    if function == 'ellipsoid':
        plt.ylim([1e-8, 1e10])
        plt.yticks(ticks=[1e-8, 1e-4, 1e0, 1e4, 1e8])
    elif function == 'discus':
        plt.ylim([1e-8, 1e6])
        plt.yticks(ticks=[1e-8, 1e-4, 1e0, 1e4])
    elif function == 'rosenbrock':
        plt.ylim([1e-8, 1e10])
        plt.yticks(ticks=[1e-8, 1e-4, 1e0, 1e4, 1e8])
    plt.xlabel("Function Evaluations")
    plt.ylabel("Objective Value")
    plt.title(function.capitalize())
    plt.show()


class Fig2(R1ES):  # to save data for Fig. 2
    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p, s, y = self.initialize(args)
        self._print_verbose_info(fitness, y[0])
        e1 = np.hstack((1, np.zeros((self.ndim_problem - 1,))))  # for similarity between p and e1
        f, p_norm, stepsize, theta = [], [], [], []  # for plotting data
        while not self._check_terminations():
            y_bak = np.copy(y)
            # sample and evaluate offspring population
            x, y = self.iterate(x, mean, p, y, args)
            self._n_generations += 1
            self._print_verbose_info(fitness, y)
            mean, p, s = self._update_distribution(x, mean, p, s, y, y_bak)
            x, mean, p, s, y = self.restart_reinitialize(args, x, mean, p, s, y, fitness)
            f.append(np.min(y))
            p_norm.append(np.linalg.norm(p))
            stepsize.append(self.sigma)
            theta.append(np.arccos(np.abs(np.dot(p, e1)) / p_norm[-1]))
        res = self._collect(fitness, y, mean)
        res['p'] = p
        res['s'] = s
        res['f'] = np.sqrt(f)  # for plotting data
        res['p_norm'] = p_norm  # for plotting data
        res['stepsize'] = stepsize  # for plotting data
        res['theta'] = theta  # for plotting data
        return res


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    # plot Fig. 2
    ndim_problem = 200
    problem = {'fitness_function': cigar,
               'ndim_problem': ndim_problem,
               'lower_boundary': -10 * np.ones((ndim_problem,)),
               'upper_boundary': 10 * np.ones((ndim_problem,))}
    options = {'fitness_threshold': 1e-8,
               'seed_rng': 2022,  # not given in the original paper
               'sigma': 20 / 3}
    r1es = Fig2(problem, options)
    results = r1es.optimize()
    xx = np.arange(len(results['f'])) + 1
    plt.figure()
    plt.yscale('log')
    plt.plot(xx, results['f'], 'b-', label='sqrt(f)', fillstyle='none')
    plt.plot(xx, results['p_norm'], 'g-', label='||p||', fillstyle='none')
    plt.plot(xx, results['stepsize'], 'k-', label='sigma', fillstyle='none')
    plt.plot(xx, results['theta'], 'r-', label='theta', fillstyle='none')
    plt.xticks(ticks=[1, 1000, 2000, 3000])
    plt.xlim(1, 3000)
    plt.yticks(ticks=[1e-9, 1e-6, 1e-3, 1e0, 1e3, 1e6])
    plt.xlabel('Generations')
    plt.ylabel('Values')
    plt.legend()
    plt.show()
    # plot Fig. 3
    ndim_problem = 1000
    for func in [ellipsoid, discus, rosenbrock]:
        problem = {'fitness_function': func,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -10 * np.ones((ndim_problem,)),
                   'upper_boundary': 10 * np.ones((ndim_problem,))}
        options = {'fitness_threshold': 1e-8,
                   'max_function_evaluations': 1e8,
                   'seed_rng': 2022,  # not given in the original paper
                   'sigma': 20.0 / 3.0,
                   'saving_fitness': 1,
                   'is_restart': False}
        r1es = Fig2(problem, options)
        results = r1es.optimize()
        pickle.dump(results, open(func.__name__ + '_' + str(ndim_problem) + '.pickle', 'wb'))
        plot(func.__name__, ndim_problem)
