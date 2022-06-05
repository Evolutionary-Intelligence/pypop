"""Repeat Fig. 2 and Fig. 3 (Ellipsoid, Discus, Rosenbrock) from the following paper:
    Li, Z. and Zhang, Q., 2018.
    A simple yet efficient evolution strategy for large-scale black-box optimization.
    IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
    https://ieeexplore.ieee.org/abstract/document/8080257
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pypop7.benchmarks.base_functions import cigar, ellipsoid, discus, rosenbrock
from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.r1es import R1ES

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
        fit = ES.optimize(self, fitness_function)
        xx, mean, p, s, y = self.initialize(args)
        fit.append(y[0])
        e1 = np.hstack((1, np.zeros((self.ndim_problem - 1,))))  # for similarity between p and e1
        f, p_norm, stepsize, theta = [], [], [], []  # for plotting data
        while True:
            y_bak = np.copy(y)  # for Line 13 in Algorithm 1
            xx, y = self.iterate(xx, mean, p, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fit.extend(y)
            if self._check_terminations():
                break
            mean, p, s = self._update_distribution(xx, mean, p, s, y, y_bak)
            self._n_generations += 1
            self._print_verbose_info(y)
            xx, mean, p, s, y = self.restart_initialize(args, xx, mean, p, s, y, fit)
            f.append(np.min(y))
            p_norm.append(np.linalg.norm(p))
            stepsize.append(self.sigma)
            theta.append(np.arccos(np.abs(np.dot(p, e1)) / p_norm[-1]))
        res = self._collect_results(fit, mean)
        res['p'] = p
        res['s'] = s
        res['f'] = np.sqrt(f)  # for plotting data
        res['p_norm'] = p_norm  # for plotting data
        res['stepsize'] = stepsize  # for plotting data
        res['theta'] = theta  # for plotting data
        return res


if __name__ == '__main__':
    # plot Fig. 2
    ndim_problem = 200
    problem = {'fitness_function': cigar,
               'ndim_problem': ndim_problem,
               'lower_boundary': -10 * np.ones((ndim_problem,)),
               'upper_boundary': 10 * np.ones((ndim_problem,))}
    options = {'fitness_threshold': 1e-8,
               'seed_rng': 2022,  # not given in the original paper
               'sigma': 20 / 3,
               'verbose_frequency': 200}
    r1es = Fig2(problem, options)
    results = r1es.optimize()
    x = np.arange(len(results['f'])) + 1
    plt.figure()
    plt.yscale('log')
    plt.plot(x, results['f'], 'b-', label='sqrt(f)', fillstyle='none')
    plt.plot(x, results['p_norm'], 'g-', label='||p||', fillstyle='none')
    plt.plot(x, results['stepsize'], 'k-', label='sigma', fillstyle='none')
    plt.plot(x, results['theta'], 'r-', label='theta', fillstyle='none')
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
                   'sigma': 20 / 3,
                   'verbose_frequency': 20000,
                   'record_fitness': True,
                   'record_fitness_frequency': 1}
        r1es = Fig2(problem, options)
        results = r1es.optimize()
        write_pickle(func.__name__, ndim_problem, results)
        plot(func.__name__, ndim_problem)
