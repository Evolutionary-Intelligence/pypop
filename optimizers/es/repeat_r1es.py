"""Repeat Fig. 2 from the following paper:
    Li, Z. and Zhang, Q., 2018.
    A simple yet efficient evolution strategy for large-scale black-box optimization.
    IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
    https://ieeexplore.ieee.org/abstract/document/8080257
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from optimizers.es.es import ES
from optimizers.es.r1es import R1ES

sns.set_theme(style='darkgrid')


class Fig2(R1ES):  # to save data for Fig. 2
    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p, s, y = self.initialize(args)
        fitness.append(y[0])
        e1 = np.hstack((1, np.zeros((self.ndim_problem - 1,))))  # for similarity between p and e1
        f, p_norm, stepsize, theta = [], [], [], []  # for plotting data
        while True:
            y_bak = np.copy(y)  # for Line 13 in Algorithm 1
            x, y = self.iterate(x, mean, p, y, args)  # sample and evaluate offspring population
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, p, s = self._update_distribution(x, mean, p, s, y, y_bak)
            self._n_generations += 1
            self._print_verbose_info(y)
            x, mean, p, s, y = self.restart_initialize(args, x, mean, p, s, y, fitness)
            f.append(np.min(y))
            p_norm.append(np.linalg.norm(p))
            stepsize.append(self.sigma)
            theta.append(np.arccos(np.abs(np.dot(p, e1)) / p_norm[-1]))
        results = self._collect_results(fitness, mean)
        results['p'] = p
        results['s'] = s
        results['f'] = np.sqrt(f)  # for plotting data
        results['p_norm'] = p_norm  # for plotting data
        results['stepsize'] = stepsize  # for plotting data
        results['theta'] = theta  # for plotting data
        return results


if __name__ == '__main__':
    # plot Fig. 2
    from benchmarks.base_functions import cigar
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
